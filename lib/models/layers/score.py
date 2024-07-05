import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from .attn import CBAMBlock,SpatialAttention,ChannelAttention
from lib.models.layers.frozen_bn import FrozenBatchNorm2d
# import sys
# sys.path.append("/home/azhong/CODE/TBSI_ConvMAE/baseline_head/TBSI-main/lib/models/layers/external/PreciseRoIPooling/pytorch/prroi_pool")
from lib.models.layers.pytorch.prroi_pool import PrRoIPool2D
# from prroi_pool import PrRoIPool2D
from timm.models.layers import trunc_normal_


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class ScoreHead(nn.Module):
    def __init__(self, inplanes=768):
        super(ScoreHead, self).__init__()
        self.conv1 = conv(inplanes, inplanes // 2)
        self.conv2 = conv(inplanes // 2, inplanes // 4)
        self.conv3 = conv(inplanes // 4, inplanes // 8)
        self.fc1 = nn.Linear(inplanes // 8 * 8 * 8, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.cls = nn.Linear(1024, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, H, W = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.contiguous().view(B, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.cls(x)
        return out


def build_score_head(hidden_dim):
    # score_head = ScoreHead(hidden_dim)
    score_head = ScoreDecoder(hidden_dim=hidden_dim)
    return score_head


class ScoreDecoder(nn.Module):
    def __init__(self, num_heads=12, hidden_dim=768, nlayer_head=3, pool_size=4):
        super().__init__()
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.score_head = MLP(hidden_dim, hidden_dim, 1, nlayer_head)
        self.scale = hidden_dim ** -0.5
        self.search_prroipool = PrRoIPool2D(pool_size, pool_size, spatial_scale=1.0)
        self.proj_q = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_k = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_v = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.proj = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))

        self.score_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # 对score_token初始化
        trunc_normal_(self.score_token, std=.02)

    # search_feat与template_feat是传入corner_head之前的特征，search_box是经过corner_head之后的归一化坐标
    def forward(self, template_online_feat, search_feat, search_box):
        """
        :search_feat: 传入corner_head之前的特征
        :template_feat: 传入corner_head之前的特征
        :search_box: 经过corner_head之后的归一化坐标
        :param search_box: with normalized coords. (x0, y0, x1, y1)
        :return:
        """
        b, c, h, w = search_feat.shape  # torch.Size([32, 768, 16, 16])
        # 将预测框坐标映射到特征图上
        search_box = search_box.clone() * w
        # bb_pool = box_cxcywh_to_xyxy(search_box.view(-1, 4))
        bb_pool = search_box.view(-1, 4)
        # Add batch_index to rois
        batch_size = bb_pool.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb_pool.device)
        # 拼接成roi形式
        target_roi = torch.cat((batch_index, bb_pool), dim=1) # 1 5

        # decoder1: query for search_box feat
        # decoder2: query for template feat
        x = self.score_token.expand(b, -1, -1) # 1 1 768
        x = self.norm1(x)
        # rearrange是为了按照某种模式重组tensor
        #将roi在search_feat截取出来并将HW展平
        search_box_feat = rearrange(self.search_prroipool(search_feat, target_roi), 'b c h w -> b (h w) c')
        # template_feat = rearrange(template_feat, 'b c h w -> b (h w) c')
        template_online_feat = rearrange(template_online_feat, 'b c h w -> b (h w) c')
        kv_memory = [search_box_feat, template_online_feat]
        for i in range(2):
            q = rearrange(self.proj_q[i](x), 'b t (n d) -> b n t d', n=self.num_heads)
            k = rearrange(self.proj_k[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)
            v = rearrange(self.proj_v[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)

            attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
            attn = F.softmax(attn_score, dim=-1)
            x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
            x = rearrange(x, 'b h t d -> b t (h d)')   # (b, 1, c)
            x = self.proj[i](x)
            x = self.norm2[i](x)
        out_scores = self.score_head(x).squeeze(-1)  # (b, 1, 1)

        return out_scores


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        res = x
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x+res)
        return x