import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import datetime
import time
import copy
from pathlib import Path
from qpth.qp import QPFunction
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import json
from dataset import SetDataManager
from options import parse_args
import clip_plot
import random
import numpy as np
_tokenizer = _Tokenizer()
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        n_ctx_vision = 2  # the number of vision context tokens
        ctx_init_flag = True
        dtype = torch.float
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        M = 4  # the number of our visual prompts
        N = 4  # the number of our text prompts
        self.M = M
        self.N = N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init_flag:
            ctx_list = ["a photo of a","this is a photo ","this is picture of","one picture of a"]
            n_ctx = len(ctx_list[0].split())
            ctx_vectors_list = []
            prompt_prefix_list = []
            
            for i in range(N):
                ctx_init = ctx_list[i].replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                ctx_vectors_list.append(ctx_vectors)
                prompt_prefix = ctx_init
                prompt_prefix_list.append(prompt_prefix)
            ctx_vision_vectors = torch.empty(M, n_ctx_vision ,768, dtype=dtype)
            nn.init.normal_(ctx_vision_vectors, std=0.02)
            ctx_vectors = torch.stack(ctx_vectors_list)
            
        else:
            ctx_vectors = torch.empty(N, n_ctx, ctx_dim, dtype=dtype)
            ctx_vision_vectors = torch.empty(M, n_ctx_vision ,768, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vision_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.ctx = nn.Parameter(ctx_vectors)  # parameters of text prompt to be learned
        self.ctx_vision = nn.Parameter(ctx_vision_vectors)  # parameters of vision prompt to be learned
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        prompt_list = []
        if ctx_init:
            for i in range(N):
                prompt_prefix = prompt_prefix_list[i]
                prompts = [prompt_prefix + " " + name + "." for name in classnames] # 100
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 100x77
                prompt_list.append(tokenized_prompts)
            tokenized_prompts = torch.cat(prompt_list)
        else:
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            tokenized_prompts = tokenized_prompts.repeat(N,1)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)
        ctx = ctx.permute(1, 0, 2, 3)  # N 100 16 512
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts
    
    def forward(self):

        ctx = self.ctx
        ctx_vision = self.ctx_vision
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        return prompts, ctx_vision  # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float
        self.N = 4
        self.n_cls = len(classnames)
        self.tradeoff = False  # whether use OT
        self.eps = 0.1
        self.max_iter = 100

        self.device = torch.device("cuda")
        self.device1 = torch.device("cuda")

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    def forward(self, image):
        b = image.shape[0]
        prompts, vision_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        image_features = self.image_encoder(image.type(self.dtype), vision_prompts)
        
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        text_features = self.text_encoder(prompts, tokenized_prompts).contiguous().view(self.N, self.n_cls, self.d)

        image_features = F.normalize(image_features, dim=2)  # N c d

        text_features = F.normalize(text_features, dim=2)
       
        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M, self.N, b*self.n_cls)
        sim = sim.permute(2, 0, 1)
        wdist = 1.0 - sim
        xx = torch.zeros(b*self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        
        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        
        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)

        logit_scale = self.logit_scale.exp()

        logits2 = logit_scale * sim_op
        return logits2


def finetune(novel_loader, n_way=5, n_support=5, n_query=15, class_name=()):
    iter_num = len(novel_loader)
    acc_all = []

    name = 'FT'
    if params.reset_head:
        name += '_H'
    test_log_file = open(os.path.join(params.output, f'{name}_{params.n_way}w_{params.n_shot}s.txt'), "w")
    print(params, file=test_log_file)

    model_clip, _ = clip.load("ViT-B/16", device="cpu")
    model_clip = clip_plot.build_model(model_clip.state_dict())

    start_time = time.time()
    for ti, (x, _) in enumerate(novel_loader):

        # prepare data
        x = x.cuda()
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (n_way*n_support, 3, H, W)
        ys = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
        xq = x[:, n_support:].reshape(-1, *x.size()[2:])  # (n_way*query, 3, H, W)
        yq = np.repeat(range(n_way), n_query)
        task_class_name = [class_name[i] for i in _[:, 0].numpy()]
        # Model
        model = CustomCLIP(task_class_name, copy.deepcopy(model_clip)).float().cuda()
       
        # Finetune
        model.train()
      
        batch_size = n_way
        support_size = n_way * n_support
        loss_fn = nn.CrossEntropyLoss().cuda()
        opt = torch.optim.AdamW(model.prompt_learner.parameters(), lr=params.ft_lr)

        for epoch in range(params.ft_epoch):
            rand_id = np.random.permutation(support_size)
            step = 0
            for j in range(0, support_size, batch_size // 4):
                if step == 0:
                    opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size // 4, support_size)]).cuda()
                x_batch = xs[selected_id]  # (batch_size, 3, 224, 224)
                y_batch = ys[selected_id]  # (batch_size)
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch) / 4
                loss.backward()
                if step == 3:
                    opt.step()
                step = (step + 1) % 4
        del opt, xs
        torch.cuda.empty_cache()

        # Test
        model.eval()
        with torch.no_grad():
            correct_this = 0
            count_this = 0
            for j in range(0, len(xq), 10):
                scores = model(xq[j:j+10])  # (n_way*query, n_way)
                _, topk_labels = scores.data.topk(1, 1, True, True)
                topk_ind = topk_labels.cpu().numpy()
                top1_correct = np.sum(topk_ind[:, 0] == yq[j:j+10])
                correct_this, count_this = correct_this + float(top1_correct), count_this + len(yq[j:j+10])
            acc = correct_this * 100. / count_this
        acc_all.append(acc)
        print('Task %d: %4.2f%%' % (ti, acc))
        print("Task %d: %4.2f%%" % (ti, acc), file=test_log_file)

        del xq, model
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
    if params.dataset == 'cub':
        with open(novel_file, 'r') as f:
            class_name = json.load(f)['label_names']
        for i in range(len(class_name)):
            class_name[i] = class_name[i].split('.')[-1].replace('_', ' ').lower()
    elif params.dataset == 'cars':
        with open(novel_file, 'r') as f:
            class_name = json.load(f)['label_names']
        for i in range(len(class_name)):
            class_name[i] = class_name[i].lower()
    elif params.dataset == 'places':
        with open(novel_file, 'r') as f:
            class_name = json.load(f)['label_names']
        for i in range(len(class_name)):
            class_name[i] = class_name[i].replace('_', ' ').lower()
    else:
        with open('./filelists/plantae/name.json', 'r') as f:
            js = json.load(f)
        d = {}
        for _ in js:
            d[int(_['id'])] = _['name']
        with open(novel_file, 'r') as f:
            class_name = json.load(f)['label_names']
        for i in range(len(class_name)):
            class_name[i] = d[int(class_name[i])].replace('_', ' ').lower()

    datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.n_way, n_support=params.n_shot, n_eposide=eposide_num)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)
    params.output = os.path.join(params.output, params.dataset)
    Path(params.output).mkdir(parents=True, exist_ok=True)
    finetune(novel_loader, n_way=params.n_way, n_support=params.n_shot, n_query=n_query, class_name=class_name)