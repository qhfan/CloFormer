from .blocks import EfficientBlock
import torch.utils.checkpoint as checkpoint
import torch
import torch.nn as nn
from typing import List

class CloLayer(nn.Module):

    def __init__(self, depth, dim, out_dim, num_heads, group_split: List[int], kernel_sizes: List[int],
                 window_size: int, mlp_kernel_size: int, mlp_ratio: int, attn_drop=0,
                 mlp_drop=0., qkv_bias=True, drop_paths=[0., 0.], downsample=True, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
                        [
                            EfficientBlock(dim, dim, num_heads, group_split, kernel_sizes, window_size,
                                mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[i])
                                for i in range(depth-1)
                        ]
                    )
        if downsample is True:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 2, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))
        else:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
