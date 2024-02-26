import copy
import os
import torch
import torch.nn as nn
import torch.optim
import datetime
import time
from pathlib import Path
from qpth.qp import QPFunction

import methods.backbone_vit_for_vpt as backbone_vit
from methods.metaoptnet import one_hot, kronecker
from dataset import SetDataManager
from options import parse_args

import random
import numpy as np
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class net(nn.Module):
    def __init__(self, encoder, n_way):
        super(net, self).__init__()
        self.n_way = n_way
        self.encoder = encoder
        self.head = nn.Linear(self.encoder.feat_dim, n_way)

    def train(self, mode=True):
        if mode:
            self.encoder.patch_embed.eval()
            self.encoder.pos_drop.eval()
            self.encoder.blocks.eval()
        else:
            for module in self.children():
                module.train(mode)

    def forward(self, x):  # (B, 3, H, W)
        x = self.encoder(x)
        scores = self.head(x)
        return scores

    def reset_protonet_head(self, xs):
        norm = 4000
        with torch.no_grad():
            feat = self.encoder(xs)
        z_p = feat.reshape(self.n_way, -1, feat.size(-1)).mean(1)
        state_dict = dict(weight=2.*z_p/norm, bias=-(z_p*z_p).sum(1)/norm)
        self.head.load_state_dict(state_dict)

    def reset_r2d2_head(self, xs, temp=20., lam=50):
        with torch.no_grad():
            support = self.encoder(xs)
        num_support, d = support.size()
        support_labels = torch.from_numpy(np.repeat(range(self.n_way), num_support//self.n_way)).cuda()  # (num_support)
        support_labels_one_hot = one_hot(support_labels, self.n_way)  # (num_support, n_way)

        ridge_sol = torch.mm(support, support.transpose(0, 1)) + lam * torch.eye(num_support).cuda()  # (num_support, num_support)
        ridge_sol, _ = torch.solve(torch.eye(num_support).cuda(), ridge_sol)
        ridge_sol = torch.mm(support.transpose(0, 1), ridge_sol)  # (d, num_support)

        weight = torch.mm(ridge_sol, support_labels_one_hot).transpose(0, 1)  # (n_way, d)
        state_dict = dict(weight=weight*temp, bias=torch.zeros(self.n_way).cuda())
        self.head.load_state_dict(state_dict)

    def reset_metaoptnet_head(self, xs, temp=20., C_reg=0.1, maxIter=15):
        with torch.no_grad():
            support = self.encoder(xs)
        num_support, d = support.size()
        support_labels = torch.from_numpy(np.repeat(range(self.n_way), num_support//self.n_way)).cuda()  # (num_support)

        kernel_matrix = torch.mm(support, support.transpose(0, 1))  # (num_support, num_support)
        id_matrix = torch.eye(self.n_way).cuda()  # (n_way, n_way)
        block_kernel_matrix = kronecker(kernel_matrix, id_matrix)  # (num_support*n_way, num_support*n_way)
        block_kernel_matrix += 1.0 * torch.eye(self.n_way * num_support).cuda()
        support_labels_one_hot = one_hot(support_labels, self.n_way)  # (num_support, n_way)

        G = block_kernel_matrix  # (num_support*n_way, num_support*n_way)
        e = -1. * support_labels_one_hot.flatten()  # (num_support*n_way)
        C = torch.eye(self.n_way * num_support).cuda()  # (num_support*n_way, num_support*n_way)
        h = C_reg * support_labels_one_hot.flatten()  # (num_support*n_way)
        A = kronecker(torch.eye(num_support).cuda(), torch.ones(1, self.n_way).cuda())  # (num_support, num_support*n_way)
        b = torch.zeros(num_support).cuda()  # (num_support, num_support)
        # G, e, C, h, A, b = [x.float() for x in [G, e, C, h, A, b]]
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())  # (1, num_support*n_way)
        qp_sol = qp_sol.reshape(num_support, self.n_way)  # (num_support, n_way)

        # weight = (qp_sol*support_labels_one_hot).transpose(0, 1) @ support  # (n_way, d)
        weight = qp_sol.transpose(0, 1) @ support  # (n_way, d)
        # the absolute value of weight is too small,
        # so we use large temp to increase it to be similar as other tunable parameters, which helps to optimize.
        # after this, we need to divide the prediction score by temp, otherwise the fine-tuning loss will be zero.
        state_dict = dict(weight=weight*temp, bias=torch.zeros(self.n_way).cuda())
        self.head.load_state_dict(state_dict)


def finetune(novel_loader, n_way=5, n_support=5, n_query=15, temp=1.):
    iter_num = len(novel_loader)
    acc_all = []
    name = 'VPT'
    if params.reset_head:
        name += '_H'
    test_log_file = open(os.path.join(params.output, name + f'_{params.n_way}w_{params.n_shot}s.txt'), "w")
    print(params, file=test_log_file)

    checkpoint = torch.load(os.path.join('./Pretrain', params.pretrain), map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']

    encoder = backbone_vit.__dict__[params.backbone]()

    # Load parameters
    msg = encoder.load_state_dict(checkpoint, strict=False)
    print(msg)

    start_time = time.time()
    for ti, (x, _) in enumerate(novel_loader):
        # Model
        model = net(copy.deepcopy(encoder), n_way).cuda()

        # prepare training data
        x = x.cuda()
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (n_way*n_support, 3, H, W)
        ys = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
        xq = x[:, n_support:].reshape(-1, *x.size()[2:])  # (n_way*query, 3, H, W)
        yq = np.repeat(range(n_way), n_query)
        if params.reset_head:
            temp = 20.
            model.reset_metaoptnet_head(xs, temp)

        # Finetune
        batch_size = n_way
        support_size = n_way * n_support

        loss_fn = nn.CrossEntropyLoss().cuda()
        parameters = [model.encoder.deep_prompt_embeddings] + list(model.head.parameters())
        opt = torch.optim.SGD(parameters, lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=params.ft_wd)

        model.train()
        for epoch in range(params.ft_epoch):
            rand_id = np.random.permutation(support_size)
            for j in range(0, support_size, batch_size):
                opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                x_batch = xs[selected_id]  # (batch_size, 3, 224, 224)
                y_batch = ys[selected_id]  # (batch_size)
                scores = model(x_batch)
                loss = loss_fn(scores/temp, y_batch)
                loss.backward()
                opt.step()

        # Test
        model.eval()
        with torch.no_grad():
            scores = model(xq)  # (n_way*query, n_way)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == yq)
            correct_this, count_this = float(top1_correct), len(yq)
            acc = correct_this * 100. / count_this
        acc_all.append(acc)
        print('Task %d: %4.2f%%' % (ti, acc))
        print("Task %d: %4.2f%%" % (ti, acc), file=test_log_file)

        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%' % (acc_mean, 1.96*acc_std/np.sqrt(iter_num)))
    print('Test Acc = %4.2f +- %4.2f%%' % (acc_mean, 1.96*acc_std/np.sqrt(iter_num)), file=test_log_file)

    print('Total time: {}'.format(total_time_str))
    print('Total time: {}'.format(total_time_str), file=test_log_file)
    test_log_file.close()


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args()
    print(params)

    image_size = 224
    eposide_num = 2000
    n_query = 15
    temp = 1.

    print('Loading target dataset!')
    novel_file = os.path.join(params.data_dir, params.dataset, 'all.json')
    datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.n_way, n_support=params.n_shot, n_eposide=eposide_num)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    params.output = os.path.join(params.output, params.dataset)
    Path(params.output).mkdir(parents=True, exist_ok=True)
    finetune(novel_loader, n_way=params.n_way, n_support=params.n_shot, n_query=n_query, temp=temp)