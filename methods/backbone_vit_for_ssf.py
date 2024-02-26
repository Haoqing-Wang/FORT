from functools import partial
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import DropPath, trunc_normal_


def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    # nn.init.normal_(scale, mean=1, std=.02)
    # nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm_layer = norm_layer

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)
        if norm_layer:
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        if self.norm_layer:
            x = ssf_ada(self.norm(x), self.ssf_scale_2, self.ssf_shift_2)
        else:
            x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim * 3)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = ssf_ada(self.qkv(x), self.ssf_scale_1, self.ssf_shift_1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        logit = (q @ k.transpose(-2, -1)) * self.scale
        attn = logit.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        x = self.proj_drop(x)
        return x, logit


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

    def forward(self, x):
        tmp, attn = self.attn(ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1))
        x = x + self.drop_path1(tmp)
        x = x + self.drop_path2(self.mlp(ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2)))
        return x, attn


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, drop_rate=0., **kwargs):
        super().__init__()
        self.feat_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(self.feat_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x, attn = blk(x)
        x = self.norm(x)
        x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        return x[:, 0], attn[:, :, :, 1:]


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model