import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import math
from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        # else:
        #     print('Unknown mode!')
        #     sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return gate


def INF(B,H,W,device):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        device = proj_query_H.device
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width,device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class CoDCT_CC(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        c2wh = dict([(64,56), (128,28), (256,14) ,(384,10), (768,7)])
        
        # self.gct = FcaBasicBlock(channels)
        self.dct = MultiSpectralAttentionLayer(channels, c2wh[channels], c2wh[channels],  reduction=16, freq_sel_method = 'top16')
        self.cc_z=CrissCrossAttention(in_dim=channels)
        self.cc_z2=CrissCrossAttention(in_dim=channels)
        self.cc_x=CrissCrossAttention(in_dim=channels)

    def forward(self, x, z, z2):
        b, c, _, _ = x.size()
        residual_x=x
        residual_z=z
        residual_z2=z2
        ws_x=self.dct(x)
        ws_z=self.dct(z)
        ws_z2=self.dct(z2)
        out_x = (ws_x+ws_z+ws_z2)/3 * x
        out_z = (ws_x+ws_z+ws_z2)/3 * z
        out_z2 = (ws_x+ws_z+ws_z2)/3 * z2
        # out_x, out_z = self.gct(x,z)
        out_x = self.cc_x(out_x) + residual_x
        out_z = self.cc_z(out_z) + residual_z
        out_z2 = self.cc_z2(out_z2) + residual_z2
        return out_x, out_z, out_z2

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual

class CBAMBlock_CoSE_SP(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size_z=5,kernel_size_x=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa_z=SpatialAttention(kernel_size=kernel_size_z)
        self.sa_x=SpatialAttention(kernel_size=kernel_size_x)

    def forward(self, x, z):
        b, c, _, _ = x.size()
        residual_x=x
        residual_z=z
        ws_x=self.ca(x)
        ws_z=self.ca(z)
        out_x = (ws_x+ws_z)/2 * x
        out_z = (ws_x+ws_z)/2 * z
        ww = self.sa_x(out_x)
        out_x=out_x*self.sa_x(out_x) + residual_x
        out_z=out_z*self.sa_z(out_z) + residual_z
        return out_x, out_z

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class CoCBAMBlock(nn.Module):
    def __init__(self, channel = 768, reduction = 16, kernel_size = 49):
        super().__init__()
        self.ca_r = ChannelAttention(channel=channel, reduction=reduction)
        self.sa_r = SpatialAttention(kernel_size=kernel_size)
        self.ca_t = ChannelAttention(channel=channel, reduction=reduction)
        self.sa_t = SpatialAttention(kernel_size=kernel_size)
        self.conv = nn.Conv2d(2*channel, channel, 1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(channel)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, R, T):
        x = torch.cat((R, T), dim=1)
        x = self.relu(self.norm(self.conv(x)))
        R_c = self.ca_r(x) * R
        T_c = self.ca_t(x) * T
        R_cs = self.sa_r(R_c) * R_c
        T_cs = self.sa_t(T_c) * T_c
        return R_cs + R, T_cs + T

class CrossAtt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, z):
        B, N_x, C = x.shape
        _, N_z, _ = z.shape
        kv = self.kv(x).reshape(B, N_x, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  
        q = self.q(z).reshape(B, N_z, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_z, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class CrossAtt(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
        
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, z):
#         B, N_x, C = x.shape
#         _, N_z, _ = z.shape
#         k = x
#         v = x
#         q = z

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N_z, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
    
class CrossAttBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = dim * mlp_ratio
        self.norm1_x = norm_layer(dim)
        self.norm1_z = norm_layer(dim)
        self.att = CrossAtt(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
    
    def forward(self, x, z):
        len_z = z.shape[1]
        # 通过互注意力算出中间值 mid.shape = z.shape
        mid = self.drop_path(self.att(self.norm1_x(x), self.norm1_z(z))) + z
        # 通过mlp进行mid与x的一层注意力，相当于out = [z_fuse, x_fuse]
        out = self.drop_path(self.mlp(self.norm2(torch.cat((mid, x), dim=1)))) + torch.cat((mid, x), dim=1)
        # z_fuse, x_fuse
        return out[:, :len_z, :], out[:, len_z:, :]



# class CoMHSA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.Hx = 16
#         self.Hz = 8
#         self.fuse_x = nn.Sequential(
#             nn.Linear(dim*2, dim),
#             nn.LayerNorm(dim),
#             nn.GELU()
#         )
#         self.fuse_z = nn.Sequential(
#             nn.Linear(dim*2, dim),
#             nn.LayerNorm(dim),
#             nn.GELU()
#         )
#         self.att_z2t_v = CrossAttBlock(dim)
#         self.att_z2t_i = CrossAttBlock(dim)
#         self.att_t2z = CrossAttBlock(dim)
        
#         self.att_t2z_v = CrossAttBlock(dim)
#         self.att_t2z_i = CrossAttBlock(dim)
#         self.att_z2t = CrossAttBlock(dim)
        

#     def forward(self, x_v, x_i):
#         B, N, C = x_v.shape
#         # 分割红外可见光embed
#         z_v = x_v[:, :self.Hz*self.Hz, :]
#         x_v = x_v[:, self.Hz*self.Hz:, :]
#         z_i = x_i[:, :self.Hz*self.Hz, :]
#         x_i = x_i[:, self.Hz*self.Hz:, :]

#         #融合红外可见光的x,z,即获得更好的搜索与模板
#         x_fuse = self.fuse_x(torch.cat((x_v, x_i), dim=2))
#         z_fuse = self.fuse_z(torch.cat((z_v, z_i), dim=2))

#         # x_fuse->z_fuse 融合后的搜索分支（x）与模板分支(z)计算互注意力并返回更新后的模板(z)与搜索分支（x）
#         z_fuse, x_fuse = self.att_t2z(x_fuse, z_fuse)
#         # z_fuse->x_fuse_v  z_fuse->x_fuse_i
#         x_fuse_v, z_fuse = self.att_z2t_v(z_fuse, x_v)    
#         x_fuse_i, z_fuse  = self.att_z2t_i(z_fuse, x_i)
        
#         # z_fuse->x_fuse
#         x_fuse, z_fuse = self.att_z2t(z_fuse, x_fuse)
#         #x_fuse->z_fuse_v  x_fuse->z_fuse_i
#         z_fuse_v, x_fuse = self.att_t2z_v(x_fuse, z_v)
#         z_fuse_i, x_fuse = self.att_t2z_i(x_fuse, z_i)
        
#         out_v = torch.cat((z_fuse_v, x_fuse_v), dim=1)
#         out_i = torch.cat((z_fuse_i, x_fuse_i), dim=1)

#         return out_v, out_i

class CoMHSA(nn.Module):
    def __init__(self, dim, drop_path_rate=0.):
        super().__init__()
        self.Hx = 16
        self.Hz = 8
        self.fuse_x = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.fuse_z = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.att_z2t_v = CrossAttBlock(dim, drop_path=drop_path_rate)
        self.att_z2t_i = CrossAttBlock(dim, drop_path=drop_path_rate)
        self.att_t2z = CrossAttBlock(dim, drop_path=drop_path_rate)
        
        self.att_t2z_v = CrossAttBlock(dim, drop_path=drop_path_rate)
        self.att_t2z_i = CrossAttBlock(dim, drop_path=drop_path_rate)
        self.att_z2t = CrossAttBlock(dim, drop_path=drop_path_rate)
        

    def forward(self, x_v, x_i):
        B, N, C = x_v.shape
        # 分割红外可见光embed
        z_v = x_v[:, :self.Hz*self.Hz, :]
        x_v = x_v[:, self.Hz*self.Hz:, :]
        z_i = x_i[:, :self.Hz*self.Hz, :]
        x_i = x_i[:, self.Hz*self.Hz:, :]

        #融合红外可见光的x,z,即获得更好的搜索与模板
        x_fuse = self.fuse_x(torch.cat((x_v, x_i), dim=2))
        z_fuse = self.fuse_z(torch.cat((z_v, z_i), dim=2))

        # x_fuse->z_fuse 融合后的搜索分支（x）与模板分支(z)计算互注意力并返回更新后的模板(z)与搜索分支（x）
        z_fuse, x_fuse = self.att_t2z(x_fuse, z_fuse)
        # z_fuse->x_fuse_v  z_fuse->x_fuse_i
        x_fuse_v, z_fuse = self.att_z2t_v(z_fuse, x_v)    
        x_fuse_i, z_fuse  = self.att_z2t_i(z_fuse, x_i)
        
        # z_fuse->x_fuse
        x_fuse, z_fuse = self.att_z2t(z_fuse, x_fuse)
        #x_fuse->z_fuse_v  x_fuse->z_fuse_i
        z_fuse_v, x_fuse = self.att_t2z_v(x_fuse, z_v)
        z_fuse_i, x_fuse = self.att_t2z_i(x_fuse, z_i)
        
        out_v = torch.cat((z_fuse_v, x_fuse_v), dim=1)
        out_i = torch.cat((z_fuse_i, x_fuse_i), dim=1)

        return out_v, out_i

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        split_attn = False
        len_t = 49
        if split_attn:
            attn_t = attn[..., :len_t].softmax(dim=-1)
            attn_s = attn[..., len_t:].softmax(dim=-1)
            attn = torch.cat([attn_t, attn_s], dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_st(nn.Module):
    def __init__(self, dim, mode, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        
        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        if self.mode == 's2t':  # Search to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_x, C
            v = x[:, lens_z:]  # B, lens_x, C
        elif self.mode == 't2s':  # Template to search
            q = x[:, lens_z:]  # B, lens_x, C
            k = x[:, :lens_z]  # B, lens_z, C
            v = x[:, :lens_z]  # B, lens_z, C
        elif self.mode=='t2t':  # Template to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_z, C
            v = x[:, lens_z:]  # B, lens_z, C
        elif self.mode=='s2s':  # Search to search
            q = x[:, :lens_x]  # B, lens_x, C
            k = x[:, lens_x:]  # B, lens_x, C
            v = x[:, lens_x:]  # B, lens_x, C
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C
        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.mode == 's2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 't2s':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 't2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 's2s':
            x = torch.cat([x, k], dim=1)

        if return_attention:
            return x, attn
        else:
            return x


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return y


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError

    return mapper_x, mapper_y


class FcaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        global _mapper_x, _mapper_y
        super(FcaBottleneck, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64,56), (128,28), (256,14) ,(384,10), (768,7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, z):
        residual = x
        residual_z = z

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.att(out)

        out_z = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(z)))))
        # out = self.relu(out_z)

        # out = self.conv3(self.relu(out_z))
        # out = self.bn3(self.conv3(self.relu(out_z)))
        out_z = self.bn3(self.conv3(self.relu(out_z)))

        if self.downsample is not None:
            residual = self.downsample(x)

        w1 = self.att(out)
        w2 = self.att(out_z)
        out = ((w1+w2) / 2) * out
        out_z = ((w1+w2) / 2) * out_z
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        out_z += residual_z
        out_z = self.relu(out_z)

        return out, out_z


class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, ):
        global _mapper_x, _mapper_y
        super(FcaBasicBlock, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64,56), (128,28), (256,14) ,(384,10), (768,5)])
        self.planes = planes
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, z):
        residual = x
        residual_z = z

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.conv1(x)
        # out = self.bn1(self.conv1(x))
        # out = self.relu(self.bn1(self.conv1(z)))

        # out = self.conv2(self.relu(self.bn1(self.conv1(z))))
        out_z = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(z)))))

        w1 = self.att(out)
        w2 = self.att(out_z)
        out = ((w1+w2) / 2) * out
        out_z = ((w1+w2) / 2) * out_z
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        out_z += residual_z
        out_z = self.relu(out_z)

        return out, out_z