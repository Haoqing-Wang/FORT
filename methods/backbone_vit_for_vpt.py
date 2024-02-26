import math
from operator import mul
from functools import partial, reduce
import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_tokens=10, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.feat_dim = self.embed_dim
        self.num_tokens = num_tokens
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(len(self.blocks), num_tokens, self.embed_dim))
        # xavier_uniform initialization
        val = math.sqrt(6./float(3*reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        del self.head

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def incorporate_prompt(self, x):
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((x[:, :1], self.deep_prompt_embeddings[:1].expand(x.size(0), -1, -1), x[:, 1:]), dim=1)
        return x

    def forward(self, x):
        x = self.incorporate_prompt(x)

        for i, blk in enumerate(self.blocks):
            if i > 0:
                x = torch.cat((x[:, :1], self.deep_prompt_embeddings[i:i+1].expand(x.size(0), -1, -1), x[:, (1+self.num_tokens):]), dim=1)
            x = blk(x)
        x = self.norm(x)

        return x[:, 0]


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
