"""
TBSI_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.layers.score import build_score_head
from lib.models.tbsi_track.vit_tbsi_care import vit_base_patch16_224_tbsi
from lib.utils.box_ops import box_xyxy_to_cxcywh,box_cxcywh_to_xyxy, box_xywh_to_xyxy


class TBSITrack(nn.Module):
    """ This is the base class for TBSITrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, transformer, box_head, score_branch, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim[2]
        self.backbone = transformer
        self.tbsi_fuse_search = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head
        self.score_branch = score_branch
        self.score_branch_template_fuse = conv(hidden_dim * 4, hidden_dim)
        self.template_fuse1 = conv(hidden_dim * 2, hidden_dim)
        self.template_fuse2 = conv(hidden_dim * 2, hidden_dim)
        self.search_fuse = conv(hidden_dim * 2, hidden_dim)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_online: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gt_bboxes = None):
        x, aux_dict = self.backbone(z=template, z2=template_online, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn)
        
        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, None)

        if gt_bboxes is None:
                gt_bboxes = box_cxcywh_to_xyxy(out['pred_boxes'].clone().view(-1, 4)) 
        cls_logits = self.forward_score(feat_last, gt_bboxes)

        out.update(aux_dict)
        out['cls_logits'] = cls_logits
        out['backbone_feat'] = x
        return out

    def forward_score0(self, cat_feature):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        search_v = cat_feature[:, num_template_token*2:num_template_token*2+num_search_token, :]
        enc_opt_v_0 = cat_feature[:, :num_template_token, :] # 初始模板
        enc_opt_v_1 = cat_feature[:, num_template_token:num_template_token*2, :] # 在线模板
        search_i = cat_feature[:,- num_search_token:,:]
        enc_opt_i_0 = cat_feature[:, -num_template_token*2 - num_search_token: -num_template_token - num_search_token, :]
        enc_opt_i_1 = cat_feature[:, -num_template_token - num_search_token: - num_search_token, :]
        enc_opt = torch.cat([enc_opt_v_0, enc_opt_v_1, enc_opt_i_0, enc_opt_i_1], dim=2) # 1 64 768*4
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, 8, 8) # 1 3072 8 8
        opt_feat = self.score_branch_template_fuse(opt_feat) # 1 768 8 8
        cls_logits = self.score_branch(opt_feat)

        return cls_logits

    def forward_score(self, cat_feature, gt_bboxes=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        search_v = cat_feature[:, num_template_token*2:num_template_token*2+num_search_token, :]
        enc_opt_v_0 = cat_feature[:, :num_template_token, :] # 初始模板
        enc_opt_v_1 = cat_feature[:, num_template_token:num_template_token*2, :] # 在线模板
        search_i = cat_feature[:,- num_search_token:,:]
        enc_opt_i_0 = cat_feature[:, -num_template_token*2 - num_search_token: -num_template_token - num_search_token, :]
        enc_opt_i_1 = cat_feature[:, -num_template_token - num_search_token: - num_search_token, :]
        template_innitial = torch.cat([enc_opt_v_0, enc_opt_i_0], dim=2) # 1 64 768*2
        template_online = torch.cat([enc_opt_v_1, enc_opt_i_1], dim=2)
        search = torch.cat([search_v, search_i], dim=2) # 1 256 1536
        template_innitial = (template_innitial.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        template_online = (template_online.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        search = (search.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = template_innitial.size()
        HW = int(HW/2)
        template_innitial_feat = template_innitial.view(-1, C, 8, 8) # 1 3072 8 8
        template_online_feat = template_online.view(-1, C, 8, 8)
        search_feat = search.view(-1, C, 16, 16)
        template_innitial_feat = self.template_fuse1(template_innitial_feat) # 1 768 8 8
        template_online_feat = self.template_fuse2(template_online_feat)
        search_feat = self.search_fuse(search_feat)
        cls_logits = self.score_branch(template_online_feat,search_feat, gt_bboxes)

        return cls_logits


    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 128
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.tbsi_fuse_search(opt_feat)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_tbsi_track_online(cfg, settings=None, training=True):
    pretrained = False
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_tbsi':
        backbone = vit_base_patch16_224_tbsi(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim[2]

    box_head = build_box_head(cfg, hidden_dim)
    score_branch = build_score_head(hidden_dim)

    model = TBSITrack(
        backbone,
        box_head,
        score_branch,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
    if settings != None:
        ckpt_path = settings.stage1_model

        model_dict = torch.load(ckpt_path, map_location='cpu')['net']
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        print('Load pretrained model from: ' + settings.stage1_model)

    return model
