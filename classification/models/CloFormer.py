from .layers import CloLayer
from .patch_embedding import PatchEmbedding
import torch
import torch.nn as nn
from typing import List

class CloFormer(nn.Module):

    def __init__(self, in_chans, num_classes, embed_dims: List[int], depths: List[int],
                 num_heads: List[int], group_splits: List[List[int]], kernel_sizes: List[List[int]],
                 window_sizes: List[int], mlp_kernel_sizes: List[int], mlp_ratios: List[int],
                 attn_drop=0., mlp_drop=0., qkv_bias=True, drop_path_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.mlp_ratios = mlp_ratios
        self.patch_embed = PatchEmbedding(in_chans, embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer != self.num_layers-1:
                layer = CloLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer+1], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer], 
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop, 
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], True, use_checkpoint)
            else:
                layer = CloLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer], 
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop, 
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], False, use_checkpoint)
            self.layers.append(layer)

        self.norm = nn.GroupNorm(1, embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes>0 else nn.Identity()

    def forward_feature(self, x):
        '''
        x: (b 3 h w)
        '''
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(self.norm(x))
        return x.flatten(1)

    def forward(self, x):
        x = self.forward_feature(x)
        return self.head(x)