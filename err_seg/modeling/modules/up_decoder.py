import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 8, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, guidance_dim, up=True):
        super().__init__()
        if up:        
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        embed_dim = out_channels
        self.textual_fusion = nn.Sequential(
            nn.Conv2d(embed_dim + guidance_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(embed_dim//8, embed_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(embed_dim//8, embed_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, img_guidance):
        T = x.shape[0] // img_guidance.shape[0]
        x = self.up_conv(x)
        img_guidance = repeat(img_guidance, 'B C H W -> (B T) C H W', T=T)
        x = torch.cat([x, img_guidance], dim=1)
        x = self.textual_fusion(x)
        return x
