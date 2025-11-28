import logging
import json
import fvcore.nn.weight_init as weight_init
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

import open_clip
from err_seg.third_party import imagenet_templates
from ..modules.decoder import Decoder


@SEM_SEG_HEADS_REGISTRY.register()
class ERRSEGHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_model_name: str,
        clip_pretrained_weights: str,
        decoder: Decoder
    ):
        super().__init__()
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load clip model
        name, pretrain = (clip_model_name, clip_pretrained_weights)
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device=self.device,)
        tokenizer = open_clip.get_tokenizer(name)
      
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        self.tokenizer = tokenizer

        # load text embeddings
        prompt_templates = ['A photo of a {} in the scene',]

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates)
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates)
        self.prompt_templates = prompt_templates

        self.decoder = decoder
        self.tokens = None
        self.cache = None

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "clip_model_name": cfg.MODEL.ENC.CLIP_MODEL_NAME,
            "clip_pretrained_weights": cfg.MODEL.ENC.CLIP_PRETRAINED_WEIGHTS,
            "decoder": Decoder(
                corr_fpn_inter_dim=cfg.MODEL.SEM_SEG_HEAD.CORR_FPN_INTER_DIM,
                corr_fpn_fusion_dim=cfg.MODEL.SEM_SEG_HEAD.CORR_FPN_FUSION_DIM,
                guidance_fpn_inter_dim=cfg.MODEL.SEM_SEG_HEAD.GUIDANCE_FPN_INTER_DIM,
                guidance_fpn_fusion_dim=cfg.MODEL.SEM_SEG_HEAD.GUIDANCE_FPN_FUSION_DIM,
                image_guidance_dims=cfg.MODEL.SEM_SEG_HEAD.IMAGE_GUIDANCE_DIMS,
                decoder_guidance_dims=cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS,
                guidance_proj_dims=cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS,
                text_guidance_dim=cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM,
                text_guidance_proj_dim=cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM,
                aggregator_scale_ratio=cfg.MODEL.SEM_SEG_HEAD.AGGREGATOR_SCALE_RATIO,
                aggregator_dim=cfg.MODEL.SEM_SEG_HEAD.AGGREGATOR_DIM,
                decoder_dims=cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS,
                sample_classes_num=cfg.MODEL.SEM_SEG_HEAD.SAMPLE_CLASSES_NUM
            )
        }

    @torch.no_grad()
    def class_embeddings(self, classnames, templates):
        zeroshot_weights = []
        for inx, classname in enumerate(classnames):
            # format prompt
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]
            # tokenize & embed
            texts = self.tokenizer(texts).to(self.device)
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        zeroshot_weights = zeroshot_weights.permute(1, 0, 2).float() # P T C -> T P C
        return zeroshot_weights

    def forward(self, img_features, proj_features, img_gudiances, gt_classes):
        text = self.class_texts if self.training else self.test_class_texts
        text = self.get_text_embeds(text)
        text_features = text.repeat(img_features.shape[0], 1, 1, 1)
        img_features = proj_features + [img_features]
        img_gudiances = img_gudiances[::-1]
        return self.decoder(img_features, text_features, img_gudiances, gt_classes)
    
    def get_text_embeds(self, classnames):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in self.prompt_templates]
                else:
                    texts = [template.format(classname) for template in self.prompt_templates]  # format with class
                texts = self.tokenizer(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            self.tokens = tokens
        else:
            tokens = self.tokens
        class_embeddings = self.clip_model.encode_text(tokens)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings
        