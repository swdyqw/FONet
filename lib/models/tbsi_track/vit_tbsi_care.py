from functools import partial

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_

from lib.utils.misc import is_main_process
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.utils.pos_utils import get_2d_sincos_pos_embed
from lib.models.layers.attn import CBAMBlock_CoSE_SP,CoDCT_CC

import numpy as np
from itertools import repeat
import collections.abc

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


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

    def forward(self, x):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = q.type(torch.float32)
        k = k.type(torch.float32)
        # 下面的代码不使用自动精度训练
        with autocast(enabled=False):
            # @表示矩阵乘法
            attn = ((q @ k.transpose(-2, -1)) * self.scale)
        if torch.isinf(attn).any():
            id_y = np.where(np.array(torch.isnan(attn).detach().cpu()) == True)
            print("Vit attn is inf!")
        attn0 = attn
        # 防止上下溢出
        max_val = torch.max(attn, dim=-1).values
        max_val = max_val.view(B, self.num_heads, N, 1)
        attn = attn - max_val
        attn_ori = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if torch.isnan(attn).any():
            id_x = np.where(np.array(torch.isnan(attn).detach().cpu()) == True)
            print("Vit attn is nan!")

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x



class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class ConvViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size_s=256, img_size_t=128, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
                 depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], in_chans=3, num_classes=1000,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = 16
        self.R_patch_embed1 = PatchEmbed(
            patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.R_patch_embed2 = PatchEmbed(
            patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.R_patch_embed3 = PatchEmbed(
            patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.R_patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.R_CoSESP1 = CoDCT_CC(embed_dim[0])
        self.R_CoSESP2 = CoDCT_CC(embed_dim[1])
        self.R_CoSESP3 = CoDCT_CC(embed_dim[2])

        

        self.I_patch_embed1 = PatchEmbed(
            patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.I_patch_embed2 = PatchEmbed(
            patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.I_patch_embed3 = PatchEmbed(
            patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.I_patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.I_CoSESP1 = CoDCT_CC(embed_dim[0])
        self.I_CoSESP2 = CoDCT_CC(embed_dim[1])
        self.I_CoSESP3 = CoDCT_CC(embed_dim[2])

        # self.template_fuse_v = nn.Conv2d(embed_dim[2]*2, embed_dim[2], 1) 
        # self.template_fuse_i = nn.Conv2d(embed_dim[2]*2, embed_dim[2], 1) 
        # self.CoMHSA = CoMHSA(embed_dim[2])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.R_blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.R_blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.I_blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.I_blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.norm = nn.LayerNorm(embed_dim[-1])

        self.grid_size_s = img_size_s // (patch_size[0] * patch_size[1] * patch_size[2])
        self.grid_size_t = img_size_t // (patch_size[0] * patch_size[1] * patch_size[2])
        self.num_patches_s = self.grid_size_s ** 2
        self.num_patches_t = self.grid_size_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim[2]), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim[2]), requires_grad=False)
        self.init_pos_embed()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # 位置嵌入
    def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                              cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, z, z2, x, **kwargs):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        ### conv embeddings for x_t
        B, Hx, Wx = x[0].shape[0], x[0].shape[2], x[0].shape[3]
        B, Hz, Wz = z[0].shape[0], z[0].shape[2], z[0].shape[3]
        # if torch.isnan(z[0]).any() or torch.isnan(z[1]).any() or torch.isnan(x[0]).any() or torch.isnan(x[1]).any():
        #     print("Input data is error!!!")

        x_v = self.R_patch_embed1(x[0])
        z_v = self.R_patch_embed1(z[0])
        z2_v = self.R_patch_embed1(z2[0])

        x_i = self.I_patch_embed1(x[1])
        z_i = self.I_patch_embed1(z[1])
        z2_i = self.I_patch_embed1(z2[0]) 
        # 错误1
        

        # 第一层嵌入
        x_v = self.pos_drop(x_v)    
        z_v = self.pos_drop(z_v) 
        x_i = self.pos_drop(x_i)    
        z_i = self.pos_drop(z_i) 

        for blk in self.R_blocks1:
            x_v = blk(x_v) 
            z_v = blk(z_v)
            z2_v = blk(z2_v)
            # torch.Size([2, 256, 64, 64])
            # torch.Size([2, 256, 32, 32])
            # torch.Size([2, 256, 32, 32])


        x_v, z_v, z2_v = self.R_CoSESP1(x_v, z_v, z2_v)

        for blk in self.I_blocks1:
            x_i = blk(x_i)
            z_i = blk(z_i)
            z2_i = blk(z2_i)
        x_i, z_i, z2_i = self.I_CoSESP1(x_i, z_i, z2_i)

        # 第二层嵌入
        x_v = self.R_patch_embed2(x_v)    
        z_v = self.R_patch_embed2(z_v)
        z2_v = self.R_patch_embed2(z2_v)
        x_v = self.pos_drop(x_v)    
        z_v = self.pos_drop(z_v)
        z2_v = self.pos_drop(z2_v)
        for blk in self.R_blocks2:    
            x_v = blk(x_v)
            z_v = blk(z_v)
            z2_v = blk(z2_v)
            # torch.Size([2, 384, 32, 32])
            # torch.Size([2, 384, 16, 16])
            # torch.Size([2, 384, 16, 16])
        x_v, z_v, z2_v = self.R_CoSESP2(x_v, z_v, z2_v)
        
        x_i = self.I_patch_embed2(x_i)    
        z_i = self.I_patch_embed2(z_i)
        z2_i = self.I_patch_embed2(z2_i)
        x_i = self.pos_drop(x_i)    
        z_i = self.pos_drop(z_i)
        z2_i = self.pos_drop(z2_i)
        for blk in self.I_blocks2:    
            x_i = blk(x_i)
            z_i = blk(z_i)
            z2_i = blk(z2_i)
        x_v, z_v, z2_v = self.I_CoSESP2(x_v, z_v, z2_v)
        # 错误2
    

        # 第三层嵌入
        x_v = self.R_patch_embed3(x_v)    
        z_v = self.R_patch_embed3(z_v)
        z2_v = self.R_patch_embed3(z2_v)
        # torch.Size([2, 768, 16, 16])
        # torch.Size([2, 768, 8, 8])
        # torch.Size([2, 768, 8, 8])

        x_v, z_v, z2_v = self.R_CoSESP3(x_v, z_v, z2_v)
        x_i = self.I_patch_embed3(x_i)    
        z_i = self.I_patch_embed3(z_i)
        z2_i = self.I_patch_embed3(z2_i)
        x_i, z_i, z2_i = self.I_CoSESP3(x_i, z_i, z2_i)

        # 模板融合
        # z_v = self.template_fuse_v(torch.cat((z_v, z2_v), dim=1))
        # z_i = self.template_fuse_i(torch.cat((z_i, z2_i), dim=1))

        x_v = x_v.flatten(2).transpose(1, 2)
        z_v = z_v.flatten(2).transpose(1, 2)
        z2_v = z2_v.flatten(2).transpose(1, 2)
        x_i = x_i.flatten(2).transpose(1, 2)
        z_i = z_i.flatten(2).transpose(1, 2)
        z2_i = z2_i.flatten(2).transpose(1, 2)

        x_v = self.R_patch_embed4(x_v)
        z_v = self.R_patch_embed4(z_v)
        z2_v = self.R_patch_embed4(z2_v)
        x_i = self.I_patch_embed4(x_i)
        z_i = self.I_patch_embed4(z_i)
        z2_i = self.I_patch_embed4(z2_i)
        
        B, C = x_v.size(0), x_v.size(-1)

        # Embedding+位置编码
        x_v = x_v + self.pos_embed_s
        z_v = z_v + self.pos_embed_t
        z2_v = z2_v + self.pos_embed_t
        x_i = x_i + self.pos_embed_s
        z_i = z_i + self.pos_embed_t
        z2_i = z2_i + self.pos_embed_t
        
        x_v = torch.cat((z_v, z2_v, x_v), dim=1)
        x_v = self.pos_drop(x_v)

        x_i = torch.cat((z_i, z2_i, x_i), dim=1)
        x_i = self.pos_drop(x_i)

        index = 0
        for i, blk in enumerate(self.blocks3):
            x_v = blk(x_v)
            x_i = blk(x_i)
        # x_v, x_i = self.CoMHSA(x_v, x_i)
        x = torch.cat([x_v, x_i], dim=1)
        aux_dict = {"attn": None}
        
        return self.norm(x), aux_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = ConvViT(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")            
            checkpoint_model = checkpoint['model']
            model_dict = {}
            for key in checkpoint_model:
                if (key.startswith("patch_embed")):
                    model_dict["R_" + key] = checkpoint_model[key]
                    model_dict["I_" + key] = checkpoint_model[key]
                    continue
                if (key.startswith("blocks1") or key.startswith("blocks2")):
                    model_dict["R_" + key] = checkpoint_model[key]
                    model_dict["I_" + key] = checkpoint_model[key]
                    continue
                if (key.startswith("fc_norm")):
                    model_dict[key[3:]] = checkpoint_model[key]
                else:
                    model_dict[key] = checkpoint_model[key]
            missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
            # print(missing_keys)
            print('Load pretrained model from: ' + pretrained)
            

    return model


def vit_base_patch16_224_tbsi(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(norm_layer=nn.LayerNorm, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model
