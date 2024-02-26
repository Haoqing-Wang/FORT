import copy
import os
import torch
import torch.nn as nn
import torch.optim
import datetime
import time
from pathlib import Path

import methods.backbone_vit as backbone_vit
import methods.backbone_swin as backbone_swin

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

    def forward(self, x):  # (B, 3, H, W)
        x, _ = self.encoder(x)
        scores = self.head(x.detach())
        return scores


def finetune(novel_loader, n_way=5, n_support=5, n_query=15):
    iter_num = len(novel_loader)
    acc_all = []

    name = 'LP'
    test_log_file = open(os.path.join(params.output, f'{name}_{params.n_way}w_{params.n_shot}s.txt'), "w")
    print(params, file=test_log_file)

    checkpoint = torch.load(os.path.join('./Pretrain', params.pretrain), map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']

    # Model
    if 'vit' in params.backbone:
        encoder = backbone_vit.__dict__[params.backbone]()
    elif 'swin' in params.backbone:
        encoder = backbone_swin.__dict__[params.backbone]()
    else:
        print('Unknown backbone!')
        return

    # Load parameters
    msg = encoder.load_state_dict(checkpoint, strict=False)
    print(msg)

    start_time = time.time()
    for ti, (x, _) in enumerate(novel_loader):
        model = net(copy.deepcopy(encoder), n_way).cuda()
        for p in model.encoder.parameters():
            p.requires_grad = False

        # prepare training data
        x = x.cuda()
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (n_way*n_support, 3, H, W)
        ys = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
        xq = x[:, n_support:].reshape(-1, *x.size()[2:])  # (n_way*query, 3, H, W)
        yq = np.repeat(range(n_way), n_query)

        # Finetune
        model.train()
        batch_size = n_way
        support_size = n_way * n_support
        loss_fn = nn.CrossEntropyLoss().cuda()
        opt = torch.optim.SGD(model.head.parameters(), lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=params.ft_wd)

        for epoch in range(params.ft_epoch):
            rand_id = np.random.permutation(support_size)
            for j in range(0, support_size, batch_size):
                opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                x_batch = xs[selected_id]  # (batch_size, 3, 224, 224)
                y_batch = ys[selected_id]  # (batch_size)
                scores = model(x_batch)
                loss = loss_fn(scores, y_batch)
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

    print('Loading target dataset!')
    novel_file = os.path.join(params.data_dir, params.dataset, 'all.json')
    datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.n_way, n_support=params.n_shot, n_eposide=eposide_num)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    params.output = os.path.join(params.output, params.dataset)
    Path(params.output).mkdir(parents=True, exist_ok=True)
    finetune(novel_loader, n_way=params.n_way, n_support=params.n_shot, n_query=n_query)