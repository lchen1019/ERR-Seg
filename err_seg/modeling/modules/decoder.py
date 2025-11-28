import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from einops import rearrange
from .aggregator import Aggregator
from .up_decoder import UpDecoder


class Decoder(nn.Module):
    def __init__(self,
        corr_fpn_inter_dim=256,
        corr_fpn_fusion_dim=128,
        guidance_fpn_inter_dim=256,
        guidance_fpn_fusion_dim=128,
        image_guidance_dims=[1024, 512, 256, 128],
        decoder_guidance_dims=(512, 256, 128),
        guidance_proj_dims=(32, 16, 8),
        text_guidance_dim=640,
        text_guidance_proj_dim=128,
        aggregator_scale_ratio=[2, 2, 2, 2],
        aggregator_dim=128,
        decoder_dims=(64, 32, 16),
        sample_classes_num=32,
        prompt_channel=1,
    ) -> None:
        super().__init__()
        self.K = sample_classes_num
        
        # corr FPN fusion
        proj_dims = [corr_fpn_inter_dim, corr_fpn_inter_dim]
        self.corr_proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(prompt_channel, dim, kernel_size=7, stride=1, padding=3),
            ) for dim in proj_dims
        ])
        self.low_corr_fusion = nn.Conv2d(corr_fpn_inter_dim, corr_fpn_fusion_dim, kernel_size=1, stride=1)
        
        # guidance FPN fusion
        self.img_guidance_projection_agg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, guidance_fpn_inter_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(inplace=True),
            ) for d in image_guidance_dims[:2]
        ])
        self.low_img_fusion = nn.Conv2d(guidance_fpn_inter_dim, guidance_fpn_fusion_dim, kernel_size=1, stride=1)
        
        # decoder gudiance projection
        self.img_guidance_projection_dec = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(inplace=True)
            ) for d, dp in zip(decoder_guidance_dims, guidance_proj_dims)
        ])
        
        # project text guidance
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.GELU(),
        )
        
        # Aggregator
        self.aggregator_layers = nn.ModuleList([
            Aggregator(
                in_channels=aggregator_dim, num_heads=4, drop_path=0.025*i, sr_ratio=sr_ratio
            ) for i, sr_ratio in enumerate(aggregator_scale_ratio)
        ])
        
        # Up Decoder
        self.up_decoder1 = UpDecoder(aggregator_dim, decoder_dims[0], guidance_proj_dims[0], up=False)
        self.up_decoder2 = UpDecoder(decoder_dims[0], decoder_dims[1], guidance_proj_dims[1], up=True)
        self.up_decoder3 = UpDecoder(decoder_dims[1], decoder_dims[2], guidance_proj_dims[2], up=True)
        
        # output head
        self.head = nn.Conv2d(decoder_dims[-1], 1, kernel_size=1)

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed_proj(self, x, corr_conv):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = corr_conv(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def up_conv(self, corr_embed, img_guidance):
        corr_embed = self.up_decoder1(corr_embed, img_guidance[0])
        corr_embed = self.up_decoder2(corr_embed, img_guidance[1])
        corr_embed = self.up_decoder3(corr_embed, img_guidance[2])
        return corr_embed
    
    def select_classes(self, corr, gt_classes, K=32):
        B, P, T, H, W = corr.shape
        corr = corr.mean(dim=1)
        top = 3 if T > 3 else T
        freq_mapping = [1, 0.1, 0.01]
        _, indices = torch.topk(corr, k=top, dim=1)
        indices += 1
      
        freq_counts = torch.zeros(B, T, dtype=torch.float32, device=corr.device)
        for b in range(B):
            for i in range(top):
                freq = torch.histc(indices[b, i].flatten(), bins=T, min=1, max=T)
                freq_counts[b, :] += freq * freq_mapping[i]
        _, indices = torch.topk(freq_counts, k=K, dim=1)
        values, _ = indices.sort(dim=1)
        
        if self.training:
            ref_values = []
            for b in range(B):
                gt = gt_classes[b]
                C = torch.tensor([x for x in values[b] if x not in gt], device=values.device)
                C = torch.cat((gt, C))
                shuffled_indices = torch.randperm(C[:K].numel())
                ref_values.append(C[:K][shuffled_indices])
            ref_values = torch.stack(ref_values)
        else:
            ref_values = []
            for b in range(B):
                C = values[b]
                shuffled_indices = torch.randperm(C.numel())
                ref_values.append(C[shuffled_indices])
            ref_values = torch.stack(ref_values)
        return ref_values

    def forward(self, img_feats, text_feats, img_guidances, gt_classes):
        B = img_feats[0].shape[0]

        corrs = [self.correlation(corr, text_feats) for corr in img_feats]
        B, P, T, H, W = corrs[0].shape
        
        # thres
        img_guidances[0][img_guidances[0] < -16] = -16
        img_guidances[0][img_guidances[0] > 16] = 16
        img_guidances[1][img_guidances[1] < -16] = -16
        img_guidances[1][img_guidances[1] > 16] = 16
        img_guidances[2][img_guidances[2] < -16] = -16
        img_guidances[2][img_guidances[2] > 16] = 16
        img_guidances[3][img_guidances[3] < -16] = -16
        img_guidances[3][img_guidances[3] > 16] = 16
        
        # select classes
        K = self.K
        indices = self.select_classes(corrs[0], gt_classes, K=K)
        corrs_select = [torch.empty((B, P, K, H, W), device= corrs[0].device),
                        torch.empty((B, P, K, H//2, W//2), device= corrs[0].device)]
        for inx in range(len(corrs_select)):
            for b in range(B):
                corrs_select[inx][b] = corrs[inx][b, :, indices[b]]
        corrs = corrs_select
        
        
        # fusion os16 & os32 corr_embed
        corr_embeds = [self.corr_embed_proj(corr, proj) for corr, proj in zip(corrs, self.corr_proj_convs)]
        [corr_embed0, corr_embed1] = corr_embeds
        corr_embed0 = rearrange(corr_embed0, 'B P T H W -> (B T) P H W')
        corr_embed1 = rearrange(corr_embed1, 'B P T H W -> (B T) P H W')
        corr_embed1 = F.interpolate(corr_embed1, size=corr_embed0.size()[2:], mode='bilinear', align_corners=False)
        corr_embed = (corr_embed0 + corr_embed1) / 2.0
        corr_embed = self.low_corr_fusion(corr_embed)
        
        # decoder gudiance projection
        os16_img_guidance_sg = img_guidances[1].clone().detach()
        img_guidances_sg = [os16_img_guidance_sg, img_guidances[2], img_guidances[3]]
        img_guidances_sg = [proj(g) for proj, g in zip(self.img_guidance_projection_dec, img_guidances_sg)]
        
        # guidance FPN fusion
        img_guidances = [proj(g) for proj, g in zip(self.img_guidance_projection_agg, img_guidances[:2])]
        os32_img_guidance, os16_img_guidance = img_guidances
        os32_img_guidance = F.interpolate(os32_img_guidance, size=os16_img_guidance.size()[2:], mode='bilinear', align_corners=False)
        fusion_img_guidance = (os32_img_guidance + os16_img_guidance) / 2.0
        fusion_img_guidance = self.low_img_fusion(fusion_img_guidance)
        
        # project text
        text_dim = text_feats.shape[-1]
        text_feats_select = torch.empty((B, K, P, text_dim), device= text_feats.device)
        for b in range(B):
            text_feats_select[b] = text_feats[b, indices[b]]
        text_feats = text_feats_select
        text_feats = text_feats.mean(dim=-2)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_guidance = self.text_guidance_projection(text_feats)
        
        # Aggreator
        corr_embed = rearrange(corr_embed, '(B T) P H W -> B P T H W', B=B)
        for inx, layer in enumerate(self.aggregator_layers):
            corr_embed = layer(corr_embed, fusion_img_guidance, text_guidance)

        # Up Decoder
        corr_embed = rearrange(corr_embed, 'B P T H W -> (B T) P H W')
        corr_embed = self.up_conv(corr_embed, img_guidances_sg)

        # head
        corr_embed = self.head(corr_embed)
        logit = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)

        # reconstruction
        B, _, H, W = logit.shape
        logit_recon = torch.ones((B, T, H, W), device=logit.device, dtype=torch.float32)
        logit_recon = logit_recon * (-100)

        for b in range(B):
            logit_recon[b, indices[b]] = logit[b].float()
        logit = logit_recon

        return logit