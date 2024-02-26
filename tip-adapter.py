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
        self.dtype = clip_model.dtype

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


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.clip = clip_model
        self.classnames = classnames
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.clip.encode_image(image.type(self.dtype))
        text = self.classnames
        text = clip.tokenize(text).cuda()
        text_features = self.clip.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
        
    def encode_image(self, image):
        image_features = self.clip.encode_image(image.type(self.dtype))
        return image_features / image_features.norm(dim=-1, keepdim=True)
        
    def encode_text(self):
        text = self.classnames
        text = clip.tokenize(text).cuda()
        text_features = self.clip.encode_text(text)
        return text_features / text_features.norm(dim=-1, keepdim=True)


def one_hot(indices, depth=20):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()  # (n_batch, m, depth) or (m, depth)
    index = indices.reshape(indices.size()+torch.Size([1]))  # (n_batch, m, 1) or (m, 1)
    if len(indices.size()) < 2:
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    else:
        encoded_indicies = encoded_indicies.scatter_(2, index, 1)
    return encoded_indicies


def finetune(novel_loader, n_way=5, n_support=5, n_query=15, class_name=()):
    iter_num = len(novel_loader)
    acc_all = []

    name = 'FT'
    if params.reset_head:
        name += '_H'
    test_log_file = open(os.path.join(params.output, f'{name}_{params.n_way}w_{params.n_shot}s.txt'), "w")
    print(params, file=test_log_file)

    model_clip, _ = clip.load("ViT-B/16", device="cpu")

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
        
        cache_keys = []
        cache_values = []
        for j in range(0, xs.size(0), 20):
            cache_key = model.encode_image(xs[j:j+20]).data
            cache_value = one_hot(ys[j:j+20])
            cache_keys += [cache_key]
            cache_values += [cache_value]
        cache_keys = nn.Parameter(torch.cat(cache_keys, dim=0), requires_grad=True)
        cache_values = torch.cat(cache_values, dim=0)
        opt = torch.optim.AdamW([cache_keys], lr=params.ft_lr)
        for epoch in range(params.ft_epoch):
            rand_id = np.random.permutation(support_size)
            for j in range(0, support_size, batch_size):
                opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                x_batch = xs[selected_id]  # (batch_size, 3, 224, 224)
                y_batch = ys[selected_id]  # (batch_size)
                alpha = 3
                beta = 1
                image_features = model.encode_image(x_batch).data
                text_features = model.encode_text().data
                clip_logits = 100. * image_features @ text_features.t()
                affinity = image_features @ cache_keys.t()
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        
                logits = clip_logits + cache_logits * alpha
                loss = loss_fn(logits, y_batch)
                loss.backward()
                opt.step()
        del opt, xs
        torch.cuda.empty_cache()

        # Test
        model.eval()
        with torch.no_grad():
            alpha = 3
            beta = 1
            image_features = model.encode_image(xq)
            text_features = model.encode_text()
            clip_logits = 100. * image_features @ text_features.t()
            affinity = image_features @ cache_keys.t()
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
            tip_logits = clip_logits + cache_logits * alpha
            
            _, topk_labels = tip_logits.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == yq)
            correct_this, count_this = float(top1_correct), len(yq)
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
        class_name = ['a photo of a %s, a type of bird.'%name for name in class_name]
    elif params.dataset == 'cars':
        with open(novel_file, 'r') as f:
            class_name = json.load(f)['label_names']
        for i in range(len(class_name)):
            class_name[i] = class_name[i].lower()
        class_name = ['a photo of a %s.'%name for name in class_name]
    elif params.dataset == 'places':
        with open(novel_file, 'r') as f:
            class_name = json.load(f)['label_names']
        for i in range(len(class_name)):
            class_name[i] = class_name[i].replace('_', ' ').lower()
        class_name = ['a photo of a %s.'%name for name in class_name]
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
        class_name = ['a photo of a %s.'%name for name in class_name]

    datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.n_way, n_support=params.n_shot, n_eposide=eposide_num)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    params.output = os.path.join(params.output, params.dataset)
    Path(params.output).mkdir(parents=True, exist_ok=True)
    finetune(novel_loader, n_way=params.n_way, n_support=params.n_shot, n_query=n_query, class_name=class_name)