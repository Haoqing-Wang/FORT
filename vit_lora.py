import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
import datetime
import time
import copy
from pathlib import Path
from qpth.qp import QPFunction

import methods.backbone_vit_for_lora as backbone_vit
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


# class Denormalise(transforms.Normalize):
#     """
#     Undoes the normalization and returns the reconstructed images in the input domain.
#     """
#     def __init__(self, mean, std):
#         mean = torch.as_tensor(mean)
#         std = torch.as_tensor(std)
#         std_inv = 1 / (std + 1e-12)
#         mean_inv = -mean * std_inv
#         super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)
#
#     def __call__(self, tensor):
#         return super(Denormalise, self).__call__(tensor.clone())
#
#
# denorm = Denormalise(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#
# def img_plot(x, name):
#     img = denorm(x).clamp(min=0., max=1.).permute(1, 2, 0)
#     img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
#     img.save(f'{name}.png')
#
#
# def img_plot_target(x, target, name):
#     target = target.unsqueeze(1).unsqueeze(3).repeat(1, 16, 1, 16).reshape(224, 224, 1).repeat(1, 1, 3)
#     x = denorm(x).clamp(min=0., max=1.).permute(1, 2, 0)  # (224, 224, 3)
#     red = torch.tensor([1., 0., 0.]).unsqueeze(0).unsqueeze(0)
#     x = x * (1. - target) + red * target
#     img = Image.fromarray((x.numpy() * 255).astype(np.uint8))
#     img.save(f'{name}.png')
#
#
# def attn_plot(attn, name):
#     attn = attn.reshape(int(attn.size(0)**0.5), int(attn.size(0)**0.5))
#     sns.heatmap(attn.numpy(), cbar=False, square=True, cmap="viridis", xticklabels=False, yticklabels=False)
#     plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0.0)
#     plt.close()
#
#
# def attn_plot_top(attn, name, p=0.95):
#     prob, ids_shuffle = torch.sort(attn, descending=True)
#     s = 0.
#     for L, x in enumerate(prob):
#         s += x.item()
#         if s > p:
#             break
#     ids_restore = torch.argsort(ids_shuffle)
#     mask = torch.ones_like(attn)
#     mask[L+1:] = 0
#     mask = torch.gather(mask, dim=0, index=ids_restore)
#     # mask = mask * attn
#     mask = mask.reshape(int(attn.size(0)**0.5), int(attn.size(0)**0.5)).bool()
#     sns.heatmap(mask.numpy(), cbar=False, square=True, xticklabels=False, yticklabels=False)
#     plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0.0)
#     plt.close()


class net(nn.Module):
    def __init__(self, encoder, n_way):
        super(net, self).__init__()
        self.n_way = n_way
        self.encoder = encoder
        self.head = nn.Linear(self.encoder.feat_dim, n_way)

        self.hooks = []
        for name, module in self.named_modules():
            if 'attn_drop' in name:
                self.hooks.append(module.register_forward_hook(self.get_attention))
            if 'blocks.11.norm1' in name:
                self.hooks.append(module.register_backward_hook(self.get_gradient))
        self.attentions = []
        self.gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_input[0].cpu())

    def forward(self, img):  # (B, 3, H, W)
        x, attn = self.encoder(img)
        scores = self.head(x)
        return scores, attn.mean(1)

    def reset_protonet_head(self, xs):
        self.encoder.eval()
        norm = 4000
        with torch.no_grad():
            feat, _ = self.encoder(xs)
        z_p = feat.reshape(self.n_way, -1, feat.size(-1)).mean(1)
        state_dict = dict(weight=2.*z_p/norm, bias=-(z_p*z_p).sum(1)/norm)
        self.head.load_state_dict(state_dict)

    def reset_r2d2_head(self, xs, temp=20., lam=50):
        self.encoder.eval()
        with torch.no_grad():
            support, _ = self.encoder(xs)
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
        self.encoder.eval()
        with torch.no_grad():
            support, _ = self.encoder(xs)
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


def get_first_comp(inp):  # cleaner
    inp[torch.isnan(inp)] = 0.
    inp = inp - inp.mean(1, keepdim=True)
    U, S, V = torch.svd(inp, some=False)
    projection = inp @ V[:, :, :1]
    return projection.squeeze()


def importance(net, inp, lab, lamb=1.):
    net.zero_grad()
    output, _ = net(inp)
    category_mask = torch.zeros(output.size()).to(output.device)
    category_mask = category_mask.scatter_(1, lab.unsqueeze(1), 1)
    logit = (output * category_mask).sum(-1).mean()
    logit.backward()
    net.zero_grad()
    attns, grads = net.attentions, net.gradients

    grad = get_first_comp(grads[0][:, 1:].cpu())

    with torch.no_grad():
        result = torch.eye(attns[0].size(-1)-1).unsqueeze(0).to(attns[0].device)  # (1, L, L)
        for attn in attns:
            attn_fused = attn.min(1)[0][:, 1:, 1:] + lamb * grad.unsqueeze(1)
            _, indices = attn_fused.topk(int(attn_fused.size(-1) * 0.9), -1, False)
            attn_fused = attn_fused.scatter_(-1, indices, 0)

            I = torch.eye(attn_fused.size(-1)).unsqueeze(0).to(attn_fused.device)
            a = (attn_fused + I) / 2.
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result
    imp = result.mean(1)

    # del hook
    del net.attentions, net.gradients
    for hook in net.hooks:
        hook.remove()

    return imp.cuda()


def imp_to_focus(imp, P):
    _, ids_shuffle = torch.sort(imp, descending=True, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    focus = torch.ones_like(imp)
    focus[:, P:] = 0
    focus = torch.gather(focus, dim=1, index=ids_restore)

    focus = imp * focus
    focus = focus * P / focus.sum(-1, keepdim=True)
    return focus


def ce_loss(y_pred, y_true):
    return - (y_true * F.log_softmax(y_pred, dim=-1)).sum(dim=-1).mean()


def finetune(novel_loader, n_way=5, n_support=5, n_query=15, temp=1.):
    iter_num = len(novel_loader)
    acc_all = []

    name = 'LoRA'
    if params.reset_head:
        name += '_H'
    if params.alpha > 0.:
        name += '_FORT'
    test_log_file = open(os.path.join(params.output, f'{name}_{params.n_way}w_{params.n_shot}s.txt'), "w")
    print(params, file=test_log_file)

    checkpoint = torch.load(os.path.join('./Pretrain', params.pretrain), map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']

    encoder = backbone_vit.__dict__[params.backbone]()

    # Load parameters
    msg = encoder.load_state_dict(checkpoint, strict=False)
    print(msg)

    start_time = time.time()
    for ti, (x, _) in enumerate(novel_loader):
        # prepare data
        x = x.cuda()
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (n_way*n_support, 3, H, W)
        ys = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
        xq = x[:, n_support:].reshape(-1, *x.size()[2:])  # (n_way*query, 3, H, W)
        yq = np.repeat(range(n_way), n_query)

        # Model
        model = net(copy.deepcopy(encoder), n_way).cuda()
        if params.reset_head:
            temp = 20.
            model.reset_metaoptnet_head(xs, temp)
            model.attentions, model.gradients = [], []

        # key location
        imp = importance(model, xs, ys)
        focus = imp_to_focus(imp, params.P)

        # plot before finetune
        # pb = 0.95
        # Path(f'./{params.dataset}').mkdir(parents=True, exist_ok=True)
        # Path(f'./{params.dataset}/support').mkdir(parents=True, exist_ok=True)
        # Path(f'./{params.dataset}/query').mkdir(parents=True, exist_ok=True)
        #
        # target = focus.reshape(focus.size(0), int(focus.size(1)**0.5), int(focus.size(1)**0.5)).cpu().bool().float()
        # for i in range(xs.size(0)):
        #     img_plot(xs[i].cpu(), f'./{params.dataset}/support/Fig{i}')
        #     img_plot_target(xs[i].cpu(), target[i], f'./{params.dataset}/support/Fig{i}_prompt')
        # for i in range(xq.size(0)):
        #     img_plot(xq[i].cpu(), f'./{params.dataset}/query/Fig{i}')
        #
        # with torch.no_grad():
        #     _, attn = model.encoder(xs)
        #     attn = attn[:, :, 0].mean(1).softmax(-1).cpu()
        #     for i in range(xs.size(0)):
        #         attn_plot_top(attn[i], f'./{params.dataset}/support/Fig{i}_top_orig', p=pb)
        #         attn_plot(attn[i], f'./{params.dataset}/support/Fig{i}_attn_orig')
        #     _, attn = model.encoder(xq)
        #     attn = attn[:, :, 0].mean(1).softmax(-1).cpu()
        #     for i in range(xq.size(0)):
        #         attn_plot_top(attn[i], f'./{params.dataset}/query/Fig{i}_top_orig', p=pb)
        #         attn_plot(attn[i], f'./{params.dataset}/query/Fig{i}_attn_orig')

        # Finetune
        model.train()
        batch_size = n_way
        support_size = n_way * n_support
        loss_fn = nn.CrossEntropyLoss().cuda()

        parameters = []
        num = 0
        for n, p in model.named_parameters():
            if ('lora_' in n) or ('head' in n):
                parameters.append(p)
                num += p.numel()
        print(f'Parameter number: {num / 1e6}M.')
        print(f'Parameter number: {num / 1e6}M.', file=test_log_file)
        opt = torch.optim.AdamW(parameters, lr=params.ft_lr)

        for epoch in range(params.ft_epoch):
            rand_id = np.random.permutation(support_size)
            for j in range(0, support_size, batch_size):
                opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                x_batch = xs[selected_id]  # (batch_size, 3, 224, 224)
                y_batch = ys[selected_id]  # (batch_size)
                focus_batch = focus[selected_id]

                scores, attn = model(x_batch)
                focus_batch = focus_batch.unsqueeze(1).expand_as(attn).reshape(-1, attn.size(-1))
                attn = attn.reshape(-1, attn.size(-1)) / params.tau
                loss = loss_fn(scores/temp, y_batch) + params.alpha * ce_loss(attn, focus_batch)
                loss.backward()
                opt.step()
        del opt
        torch.cuda.empty_cache()

        # plot after finetune
        # with torch.no_grad():
        #     name = 'our' if params.alpha > 0. else 'ft'
        #     _, attn = model.encoder(xs)
        #     attn = attn[:, :, 0].mean(1).softmax(-1).cpu()
        #     for i in range(xs.size(0)):
        #         attn_plot_top(attn[i], f'./{params.dataset}/support/Fig{i}_top_{name}', p=pb)
        #         attn_plot(attn[i], f'./{params.dataset}/support/Fig{i}_attn_{name}')
        #     _, attn = model.encoder(xq)
        #     attn = attn[:, :, 0].mean(1).softmax(-1).cpu()
        #     for i in range(xq.size(0)):
        #         attn_plot_top(attn[i], f'./{params.dataset}/query/Fig{i}_top_{name}', p=pb)
        #         attn_plot(attn[i], f'./{params.dataset}/query/Fig{i}_attn_{name}')

        # Test
        model.eval()
        with torch.no_grad():
            scores, _ = model(xq)  # (n_way*query, n_way)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == yq)
            correct_this, count_this = float(top1_correct), len(yq)
            acc = correct_this * 100. / count_this
        acc_all.append(acc)
        print('Task %d: %4.2f%%' % (ti, acc))
        print("Task %d: %4.2f%%" % (ti, acc), file=test_log_file)

        del xq, xs, model
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