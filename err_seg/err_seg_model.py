# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange

@META_ARCH_REGISTRY.register()
class ERRSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        crop_size: Tuple[float],
        clip_finetune: str,
        in_features: Tuple[str],
        ignore_value: int,
    ):
        super().__init__()
        self.sem_seg_head = sem_seg_head
        self.size_divisibility = size_divisibility
        self.in_features = in_features
        self.ignore_value = ignore_value

        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.set_clip_fine_tuning(clip_finetune)
        self.clip_resolution = crop_size
    
    @classmethod
    def from_config(cls, cfg):
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "crop_size": cfg.INPUT.CROP.SIZE,
            "in_features": cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        }
    
    @property
    def device(self):
        return self.clip_pixel_mean.device
    
    def set_weight_strategy(self):
        for name, params in self.sem_seg_head.clip_model.named_parameters():
            # CLIP visual encoder
            if "visual" in name:
                if 'gamma' in name:
                    params.requires_grad = True
                elif 'mlp' in name or 'conv_dw' in name or 'down_sampling' in name:
                    if 'weight' in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
            # CLIP text Encoder
            elif "transformer" in name:
                if "in_proj_weight" in name or 'c_fc.weight' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

    def set_full_strategy(self):
        for name, params in self.sem_seg_head.clip_model.named_parameters():
            params.requires_grad = True

    def set_clip_fine_tuning(self, clip_finetune):
        if clip_finetune == 'weight':
            self.set_weight_strategy()
        elif clip_finetune == 'full':
            self.set_full_strategy()
        else:
            raise NotImplementedError(f"clip_finetune {clip_finetune} is not implemented")


    def forward(self, batched_inputs):
        if not self.training:
            self.size_divisibility = -1
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        image_size = (images[0].shape[1], images[0].shape[2])
        
        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        _, _, H, W = clip_images.tensor.shape
        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        clip_features = self.sem_seg_head.clip_model.encode_image(clip_images_resized, dense=True)
        
        if self.training:
            gt_classes = []
            for x in batched_inputs:
                target = x["sem_seg"].to(self.device)
                tmp = torch.unique(target, return_counts=False)
                tmp = tmp[tmp != self.ignore_value]
                gt_classes.append(tmp)
        else:
            gt_classes = None
        
        clip_vis_dense = clip_features["clip_vis_dense"]
        img_gudiances = [v for k,v in clip_features.items() if k in ["os4", "os8", "os16", "os32"]]
        proj_features = [v for k,v in clip_features.items() if k in ["os16_dense"]]
        outputs = self.sem_seg_head(clip_vis_dense, proj_features, img_gudiances, gt_classes)
        
        if self.training:
            losses = {}
            if not isinstance(outputs, list):
                outputs = [outputs]
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            num_classes = outputs[0].shape[1]
            mask = targets != self.ignore_value
            
            for i, output_ in enumerate(outputs):
                output_ = F.interpolate(output_, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
                output_ = output_.permute(0, 2, 3, 1)
                _targets = torch.zeros(output_.shape, device=self.device)
                _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
                _targets[mask] = _onehot
                output_ = output_.float()
                loss = F.binary_cross_entropy_with_logits(output_, _targets)
                losses.update({f"loss_sem_seg_{i}" : loss})
            return losses
        else:
            if isinstance(outputs, tuple):            
                outputs = outputs[0].sigmoid()
            else:
                outputs = outputs.sigmoid()
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])
            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results
