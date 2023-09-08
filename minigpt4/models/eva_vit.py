# Based on EVA, BEIT, timm and DeiT code bases
# https://github.com/baaivision/EVA
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from minigpt4.common.dist_utils import download_cached_file


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    1、划分图像为固定大小的patches
    2、为每个patch学习一个固定长度的向量表示
    3、输出图像的patch序列,为后续的Transformer Encoder做准备
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # img_size -> (height img_size,width img_size)
        patch_size = to_2tuple(patch_size)  # patch_size -> (patch_size,patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 计算patch的数量
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 每个patch的形状
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # 进行patch化的卷积操作

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape  # batch_size, channels, height,width
        # FIXME look at relaxing size constraints
        # 检查输入图像大小是否符合要求
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):
    """
    解释一下Vision Transformer中相对位置编码的实现:

    在图像里,每个pixel可以看成是一个向量,就是输入的特征图中的一个向量。
    对于注意力机制来说,需要计算每个pixel向量和其他所有pixel向量之间的关系。
    但是不同位置的pixel其实Encoding的信息不一样,所以需要加入位置的信息。
    Vision Transformer这里引入了相对位置编码,就是每个pixel会编码自己相对于其他pixel的相对位置信息。
    具体来说,给每个pixel增加两个值:
    行号row:表示相对于第一行的偏移量
    列号col:表示相对于第一列的偏移量
    然后给row和col每个位置编码一个可学习的向量,其长度等于transformer的hidden size。
    在计算注意力时,将每个pixel的row向量和col向量分别加到query向量和key向量上。
    这样注意力就可以感知不同位置的pixel之间的相对位置关系,从而提升模型对图像的建模能力。
    经过训练,这些位置编码向量可以自动学习到合理的值,来控制不同相对位置pixel之间的关系。
    所以位置编码为注意力机制提供了额外的位置信息,是Vision Transformer有效处理图像的关键。
    """

    def __init__(self, window_size, num_heads):
        """

        """
        super().__init__()
        self.window_size = window_size
        # 计算相对位置索引总数
        # (2 * window_size[0] - 1)表示了在窗口内横向索引的范围。这个范围考虑了窗口的高度，通过乘以2再减去1来计算，这是因为索引从0开始。
        # (2 * window_size[1] - 1)表示了在窗口内纵向索引的范围。这个范围考虑了窗口的宽度，同样通过乘以2再减去1来计算。
        # 添加额外的索引：在这个计算中，还添加了额外的 3 个索引，分别用于表示特殊情况：
        #
        # 第一个额外索引（self.num_relative_distance - 3）表示 "cls to token" 的相对位置。
        # 第二个额外索引（self.num_relative_distance - 2）表示 "token to cls" 的相对位置。
        # 第三个额外索引（self.num_relative_distance - 1）表示 "cls to cls" 的相对位置。
        # cls to token & token 2 cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 存储相对位置编码的参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH


        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # coords用于后续计算相对位置编码，以捕获窗口内不同位置之间的相对坐标关系
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # shape : (2, Wh, Ww) 这一行将 coords_h 和 coords_w 组合成一个坐标网格 coords
        # 这一行代码将之前创建的坐标网格张量 coords 进行了展平操作，将形状为 (2, Wh, Ww) 的张量变成了形状为 (2, Wh*Ww) 的张量
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 这一行计算了相对坐标（relative_coords），它表示了每个位置与其他位置之间的相对坐标差异。
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww # 形状变成了 (2, Wh*Ww, Wh*Ww)，其中第一个维度表示坐标维度（高度和宽度），第二个和第三个维度表示位置之间的相对坐标。
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2 # 重新排列了相对坐标 relative_coords 的维度顺序，以将高度、宽度和坐标维度放在最后。
        # 这两行代码分别将相对坐标的高度和宽度维度上的值加上了 window_size 中的对应值。这是为了将坐标从以窗口左上角为原点的坐标系（通常从0开始）转换为以窗口左上角为坐标0的坐标系。
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0  #
        relative_coords[:, :, 1] += window_size[1] - 1
        # 这一行将相对坐标的高度维度上的值乘以 2 * window_size[1] - 1。这是为了将坐标映射到一个较大的范围内，以便能够容纳窗口内不同位置的相对位置关系。
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        # 用于存储相对位置索引。
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        # 这一行将之前计算的相对坐标 relative_coords 中的每个位置之间的相对坐标和存储到 relative_position_index 张量中。这样，relative_position_index 中的每个元素表示了不同位置之间的相对位置索引。
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 分别将特殊情况的相对位置索引设置为特定的值 cls to token & token 2 cls & cls to cls
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        # self.relative_position_index.view(-1) 将相对位置索引张量展平为一维，以便从权重表中查找相应的相对位置编码权重。
        relative_position_index_tensor = self.relative_position_index.view(-1)
        # self.relative_position_bias_table[self.relative_position_index.view(-1)] 使用相对位置索引查找相对位置编码权重。
        relative_position_bias_table_weight = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        # .view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1) 将查找到的权重重塑为形状为 (Wh*Ww, Wh*Ww, nH) 的张量，其中 Wh*Ww 表示窗口内的位置数，nH 表示注意力头的数量。
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww 这一行重新排列 relative_position_bias 张量的维度，以匹配多头注意力的要求。


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, use_checkpoint=False):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  # 内部其实就是一个卷积，用卷积提取特征
        num_patches = self.patch_embed.num_patches
        # 在patch embedding后的序列中,添加一个class token,用来表示整张图片的特征。这个token和NLP中的[CLS] token类似,后续分类时用这个token的特征进行分类。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # cls token

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # 绝对位置编码，加1是因为要额外添加class token
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            # 相对位置编码
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape,
                                                     num_heads=num_heads)  # 相对位置编码
        else:
            self.rel_pos_bias = None
        self.use_checkpoint = use_checkpoint
        # 这行代码的作用是实现 stochastic depth,即随机丢弃 Block 的技巧。
        # torch.linspace生成从0到drop_path_rate之间均匀间隔的depth个值。[0,drop_path_rate*1,drop_path_rate*2,....]
        # x.item()将tensor转换为python number。
        # 因此dpr是一个含有depth个dropout probability的list,这些probability随着layer深度线性增大。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        # 构建一个dropout 的 Block list 随着层数增深,drop_path probability逐渐增大。
        # 深层的Block会随机丢弃,有效防止过拟合,使模型对噪声更加鲁棒。
        # 但浅层的Block不丢弃,保证足够的信息传递到深层。
        # 这是一个有效的正则化技巧,既防止过拟合又保证性能,已被证实对Transformer类模型很有效。
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        #         self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        #         self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        #         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            # 使用截断正态分布进行初始化。使位置编码的值近似满足正态分布,但被限制在2个标准差之内,避免初始化取值过大。
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        #         if isinstance(self.head, nn.Linear):
        #             trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    #         if isinstance(self.head, nn.Linear):
    #             self.head.weight.data.mul_(init_scale)
    #             self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        """
        针对每一个transformer block里的attn和mlp模块,对权重进行缩放初始化。
        可以使得每一层的参数方差随着层数的增加而逐渐变小。
        初始化方式有利于信息在网络中间层的传播。
        """
        def rescale(param, layer_id):
            """
            通过param.div_(math.sqrt(2.0 * layer_id))实现缩放。
            """
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 对Linear层的weight用截断正态分布初始化,std=0.02。
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 对Linear层的bias初始化为0。
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 对LayerNorm层的bias初始化为0, weight初始化为1。
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """
        重置 分类头
        """
        self.num_classes = num_classes
        # nn.Identity()是一个在PyTorch中常用的层,它不会对输入进行任何操作,只是简单地将输入返回。
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # patch_embed提取patch特征
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        # 加入cls_token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # 将cls 添加在第一个位置
        x = torch.cat((cls_tokens, x), dim=1)
        # 和位置编码相加
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
        return x

    #         x = self.norm(x)

    #         if self.fc_norm is not None:
    #             t = x[:, 1:, :]
    #             return self.fc_norm(t.mean(1))
    #         else:
    #             return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        #         x = self.head(x)
        return x

    def get_intermediate_layers(self, x):
        """
        可以用来提取block内部的特征,对模型进行诊断或finetune。
        """
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            features.append(x)

        return features


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    #         if isinstance(l, (nn.MultiheadAttention, Attention)):
    #             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
    #                 tensor = getattr(l, attr)
    #                 if tensor is not None:
    #                     tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)


def create_eva_vit_g(img_size=224, drop_path_rate=0.4, use_checkpoint=False, precision="fp16"):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=1408 // 88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
    )
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")
    interpolate_pos_embed(model, state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    #     print(incompatible_keys)

    if precision == "fp16":
        #         model.to("cuda")
        convert_weights_to_fp16(model)
    return model
