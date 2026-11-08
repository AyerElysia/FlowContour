# -*- coding: utf-8 -*-
"""Checkpoint-compatible Flow + CR-MoE denoiser used by the mainline.

官方主线使用的检查点兼容流匹配 (Flow Matching) + 条件路由混合专家 (CR-MoE) 去噪器。
本模块是 FlowContour 系统中唯一定义且受支持的去噪器实现。它完全保留了已签名发布的
官方源配置的参数命名与计算图流向，确保检查点可以通过 strict=True 加载：

核心架构设计：
* 256 维点标记 (Point Tokens)：每个轮廓离散点表示为 256 维向量
* 6 层 DiT 块架构：自注意力 (Self-Attention) -> 图像交叉注意力 (Cross-Attention) -> SwiGLU 前馈网络
* 上下文交替机制：奇偶层交替感知全局 Perceiver 压缩上下文与局部轮廓采样特征上下文
* 双重条件注入：扩散内层时间步 t 与外层阶段演化进度 s (0.0 大范围演化, 1.0 精细微调) 的联合 adaLN 调制
* 轮廓级路由路径 (Contour-level Route Path)：为第 2/4/6 块 (Python 索引 1/3/5) 提供路由决策
* 共享专家 + 残差专家：在上述 MoE 块中采用常开的共享 SwiGLU FFN + E4 Top-2 (4选2) 残差专家组合
* 稠密残差位移预测头 (Dense Residual Displacement Head)：线性预测器配合深层残差 MLP

主线严格统一：本模块不包含 token 级零碎路由或逐块独立路由器，避免碎片化与不稳定性。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """应用自适应归一化调制 (Adaptive Normalization Modulation / adaLN)。

    根据条件嵌入预测出的缩放因子 (scale) 与平移因子 (shift) 对特征进行线性变换：
        output = x * (1 + scale) + shift

    Args:
        x: 输入特征张量，形状为 [B, N, C] 或类似形式。
        shift: 平移张量，形状为 [B, C]。
        scale: 缩放张量，形状为 [B, C]。

    Returns:
        torch.Tensor: 调制后的特征张量，保持与 x 相同形状。
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """RMS 均方根归一化层 (Root Mean Square Normalization)。

    与 LayerNorm 相比去除了均值偏移计算，计算速度更快且在 Transformer 中表现优异。
    参数名 weight 与签名检查点精确对齐，保证权重加载兼容。

    Args:
        dim: 归一化的特征维度大小。
        eps: 防止除以零的微小数值稳定性常数，默认 1e-6。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        # 可学习的通道缩放仿射参数
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 RMS 归一化：x / sqrt(mean(x^2) + eps) * weight。"""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """稠密门控线性单元前馈网络 (Swish Gated Linear Unit)。

    作为每个轮廓 DiT 块的基础前馈层 (FFN)，结构为：
        SwiGLU(x) = w2(silu(v(x)) * w1(x))
    其中隐藏层维度约为 input_dim * 8 / 3，并向上对齐到 64 的倍数以提升 GPU 访存效率。

    Args:
        dim: 输入与输出特征维度。
        hidden_dim: 隐藏层特征维度，若为 None 则自动按 8/3 比例计算并对齐。
        dropout: 前馈层 Dropout 丢弃率。
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # 默认按 LLaMA 惯用方式将中间维度设为输入维度的 8/3
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # 对齐到 64 的整数倍
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        # 门控分支与数值变换分支
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        # 降维投影分支
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播计算门控线性激活。"""
        return self.dropout(self.w2(F.silu(self.v(x)) * self.w1(x)))


class CyclicRoPE1D(nn.Module):
    """闭合轮廓 1D 循环旋转位置编码 (Cyclic Rotary Position Embedding)。

    专门针对闭合多边形轮廓设计的相对位置编码。
    将闭合轮廓上的离散点位置映射到圆周弧度角 [0, 2pi) 上：
        positions = i * (2pi / num_points)
    从而使点 0 与点 (num_points - 1) 在圆周上自然相邻，完全保持轮廓的闭合拓扑性质。

    Args:
        head_dim: 单个注意力头的特征维度，必须为偶数。
        num_points: 轮廓离散点数，默认为协议规定的 128 点。
    """

    def __init__(self, head_dim: int, num_points: int = 128) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_points = int(num_points)
        # 计算 RoPE 基础几何频率序列 1 / (10000 ^ (2i / d))
        freqs = 1.0 / (
            10000.0
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("freqs", freqs)

    def _build_cache(
        self,
        num_points: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构建圆周角度并在频域展开的 cos 与 sin 缓存表。"""
        # 将轮廓点索引线性映射为圆周角度 (0 到 2*pi)
        positions = (
            torch.arange(num_points, device=device, dtype=torch.float32)
            * (2.0 * math.pi / num_points)
        )
        # 外积计算每个位置在各频率基上的旋转角度
        angles = torch.outer(positions, self.freqs.to(device))
        return angles.cos(), angles.sin()

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        """对注意力特征张量应用循环旋转位置编码。

        Args:
            x: 形状为 [batch_size, num_heads, num_points, head_dim] 的 Q 或 K 张量。

        Returns:
            torch.Tensor: 经旋转位置编码后的张量，保持原形状。
        """
        num_points = int(x.shape[2])
        cos_cache, sin_cache = self._build_cache(num_points, x.device)
        # 将 head 维度交错划分为偶数与奇数分量进行 2D 平面旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos_cache[:num_points].unsqueeze(0).unsqueeze(0)
        sin = sin_cache[:num_points].unsqueeze(0).unsqueeze(0)
        # 旋转矩阵乘法: [cos, -sin; sin, cos] * [x_even; x_odd]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # 重新交错合并回原 head_dim 维度
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间位置编码 (Sinusoidal Time Embedding)。

    用于将扩散流匹配的连续时间标量 t 或外层阶段连续标量 s 编码为稠密高维向量。
    频率分布范围跨越 1 到 10000，使网络能敏锐区分不同时刻的演化动态。

    Args:
        dim: 输出的正弦嵌入维度。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """计算输入时刻 t 的正弦和余弦拼接嵌入。"""
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device) * -scale
        )
        embedding = t.unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat(
            (torch.sin(embedding), torch.cos(embedding)),
            dim=-1,
        )
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1), "constant", 0)
        return embedding


class PerceiverCompressor(nn.Module):
    """把"一整张大特征图"压缩成"一小把摘要卡片"的模块。

    ===== 先打个比方 =====
    想象你手里有一份 128 页×128 页的体检报告（每页 256 行数据），
    信息量太大。现在你要给 256 位"专家"每人发一张摘要卡，
    每位专家只关心某一类信息（比如"骨盆在左下角吗""脊柱整体是弯的吗"）。
    做法是：让这 256 位专家各自把整份报告翻一遍，每人只记下自己最关心的
    那部分内容，写成一张摘要卡。最后你手里就有 256 张卡，代表整份报告的
    精华，而不是 16384 页原始数据。

    这个类干的就是这件事：
    - "专家" = 256 个可学习的查询向量（queries）
    - "翻报告" = 交叉注意力（每个 query 对所有图像格子算相似度并加权求和）
    - "摘要卡" = 输出的 256 个全局上下文向量

    ===== 输入输出长什么样 =====
    输入 image_feat：形状 [B, 256, 128, 128]
        B    = 一次处理几张图（batch）
        256  = 每个格子上有 256 个数字（通道）
        128,128 = 图的格子数（高、宽）
        可以理解为 B 张 128×128 的表格，每格存 256 个数字

    输出：形状 [B, 256, 256]
        B    = 同样是那几张图
        256  = 256 张摘要卡
        256  = 每张卡有 256 个数字
        注意：输出已经**没有"高"和"宽"这两个维度了**，
        它不是一个"更小的图"，而是"一叠卡片"。

    ===== 为什么要压缩 =====
    后面有 128 个轮廓点，每个点都想"看整张图"来修正自己的位置。
    如果不压缩，每个点要跟 16384 个格子逐一比较，而且 Flow Matching
    推理要重复算 8 次左右，非常慢。压缩成 256 张卡之后，每个点只需要
    看 256 张卡，跟原图多大无关，速度稳定。

    Args:
        in_dim: 输入每个格子有几个数字（通道数），默认 256。
        out_dim: 每张摘要卡有几个数字，默认 256。
        num_queries: 做几张摘要卡（也就是几个"专家"），默认 256。
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 256,
        num_queries: int = 256,
    ) -> None:
        super().__init__()

        # ① 一个"翻译层"：把输入的 256 个数字转换成 256 个数字。
        # nn.Linear 内部就是一个可学习的矩阵，它把 256 个数重新组合成新的 256 个数。
        # 这里输入输出都是 256，看着像"没变化"，但矩阵里的数值是学出来的，
        # 相当于给原特征做一次有意义的重新编码。
        # 它作用在"每个格子"上，格子与格子之间互不干扰（所以不是卷积）。
        self.input_proj = nn.Linear(in_dim, out_dim)

        # ② 256 个"专家"（摘要卡模板）。
        # nn.Embedding 本质上就是一张"可学习的数字表"，形状是 [256 行, 256 列]。
        # 平时它的用法是"给行号、取那一行"，但这里我们不按行号取，
        # 而是把整整 256 行都拿出来用——每一行就是一位专家的初始想法。
        # 这些数字一开始是随机的，训练过程中会慢慢被调成有用的"提问方式"。
        self.queries = nn.Embedding(num_queries, out_dim)

        # ③ 交叉注意力：让"专家"去"翻报告"。
        # 简单说，它做三件事：
        #   - 拿每个 query 跟每个图像格子算相似度（越像权重越大）
        #   - 用这些权重把图像格子的内容加权求和
        #   - 得到该 query 的"观后感"
        # 参数说明：
        #   embed_dim=256：每个向量 256 个数字
        #   num_heads=8  ：把 256 拆成 8 组、每组 32 个数字并行算，
        #                  不同组可以关注不同类型的关系（类似多个角度看同一份数据）
        #   batch_first=True：告诉它数据第一维是"第几张图"，
        #                  否则 PyTorch 默认以为第一维是"第几个词"，会算错
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            batch_first=True,
        )

        # ④ 归一化层：把算出来的数字"拉回正常范围"。
        # 神经网络里数字容易越算越大或越算越小，LayerNorm 会把每行的
        # 256 个数字重新缩放成均值 0、波动幅度 1，让训练更稳定、不爆炸。
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        """真正执行压缩的那一步（每次前向传播都会调用）。

        Args:
            image_feat: 输入的大特征图，形状 [B, C, H, W]
                （B 张图、每格 C 个数字、高 H、宽 W）。

        Returns:
            形状 [B, num_queries, out_dim] 的摘要卡集合。
            不管输入图是 64×64 还是 256×256，输出的卡片数量都一样，
            这就是"和分辨率无关"的意思。
        """
        # 取出"这一批有几张图"，后面要把专家复制这么多份，每张图各配一套
        batch_size = int(image_feat.shape[0])

        # ===== 第一步：把"图"摊平成"一长串格子" =====
        # flatten(2)：把 (高, 宽) 两个维度合并成一个维度，
        #             [B, 256, 128, 128] → [B, 256, 16384]
        # transpose(1, 2)：把"数字"这一维挪到最后，
        #             [B, 256, 16384] → [B, 16384, 256]
        # 结果可以读成：这一批图里，每张图有 16384 个格子，每格 256 个数字。
        # （注意：摊平后"谁在谁旁边"这个位置关系就丢了，
        #   只留下"每个格子的内容是什么"。本项目能接受，因为上游 MoonViT
        #   特征本身已经带着位置信息。）
        image_tokens = image_feat.flatten(2).transpose(1, 2)

        # ===== 第二步：把每格的数字过一遍"翻译层" =====
        # 逐格独立处理，把 256 个数字重新编码成 256 个数字，内容更"好用"。
        image_tokens = self.input_proj(image_tokens)

        # ===== 第三步：把 256 位"专家"复制成一批 =====
        # self.queries.weight 形状是 [256, 256]（256 位专家 × 每人 256 个数字）
        # unsqueeze(0) 在最前面加一维 → [1, 256, 256]
        # expand(batch_size, ...) 把它"假装"复制成 [B, 256, 256]
        #   注意 expand 是"假复制"，不真的占内存，只是让 B 份共享同一块数据。
        #   因为这一步之后没有任何人修改它，所以安全。
        queries = self.queries.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # ===== 第四步：每一位专家去"翻报告"，写出一张摘要卡 =====
        # 三个输入的含义：
        #   query：256 位专家（提问方）
        #   key  ：16384 个格子的内容（被检索的索引）
        #   value：同上（真正被加权的数据）
        # 输出 compressed 形状 [B, 256, 256]：每位专家各得 256 个数字的观后感。
        # 这一步是整个模块唯一"信息浓缩"的地方，也是最耗时的部分。
        # 第二个返回值是注意力权重（谁看了谁多少），这里用不上，用 _ 丢掉。
        compressed, _ = self.cross_attn(
            query=queries,
            key=image_tokens,
            value=image_tokens,
        )

        # ===== 第五步：把"观后感"和"专家原本的想法"相加，再归一化 =====
        # 为什么要加 queries（这叫"残差连接"）：
        #   如果只保留注意力输出，可能出现"所有专家写出来的摘要都差不多"，
        #   加上各自原本的想法后，专家之间就能保持差异，各管一摊。
        # 最后 norm 把数字缩放到正常范围，交给下游使用。
        return self.norm(compressed + queries)


class SeparatePointEmbedding(nn.Module):
    """【给轮廓上的每个点做一张"身份证"】

    这个模块要干的事情，用一句话说：
    把"这个点在哪儿"和"这个点周围长什么样"分别整理成两份资料，
    再拼成一张完整的点身份证，交给后面的网络使用。

    打个比方——你要描述一栋楼的位置：
      - 第一份资料是"门牌号"（坐标 x, y）。信息很短，就两个数字。
      - 第二份资料是"周边街景照片"（从这个点所在的图像位置采到的视觉特征）。
        信息很丰富，有 256 个数字。

    所以两张资料的"厚度"天生不对等。如果直接把门牌号和街景照片订在一起，
    门牌号那 2 个数字会被 256 个数字彻底淹没，后面的网络根本注意不到位置。
    因此这里要做的关键动作是：先把门牌号"扩写"成 64 个数字，
    让它有足够的分量，再和压缩成 192 个数字的街景照片拼在一起，总共 256。

    最终每个点拿到的东西长这样：
        64 维（在哪儿） + 192 维（周围长什么样） = 256 维完整描述

    Args:
        state_dim: 每个点最终要输出多少维（默认 256）。后面网络就按这个宽度收。
        feature_dim: 采进来的图像特征原本多少维（默认 256）。
    """

    def __init__(self, state_dim: int = 256, feature_dim: int = 256) -> None:
        super().__init__()
        # ── 第一步：把 256 维这个总预算，按 1:3 分给两份资料 ──
        # 为什么是 1:3 而不是 1:1？
        #   坐标只有 2 个原始数字，再怎么扩写，有效信息也就那么多，
        #   给它 1/4（64 维）已经很舍得。
        #   街景特征本来就有 256 个数字，内容密集，自然该拿大头 3/4（192 维）。
        # 为什么两者必须严格加起来等于 state_dim？
        #   因为最后要把两份资料拼成一条，如果总长度不是 256，
        #   后面网络按 256 去接就会对不上，直接报形状错误。
        coordinate_dim = state_dim // 4
        feature_embedding_dim = state_dim - coordinate_dim

        # ── 第二步：搭"门牌号扩写器"（2 维 → 64 维 → 64 维）──
        # nn.Linear(2, 64)：把只有 2 个数字的坐标，摊开成 64 个数字。
        #   想象成：原本你只知道"东经 120、北纬 30"，
        #   扩写后变成一整套描述——偏左多少、偏上多少、离中心多远……
        #   这些新数字不是人规定的，是训练时网络自己学会的。
        # nn.SiLU()：一个"弯一弯"的函数，让变换不是死板的直线关系。
        #   如果只用 Linear 叠 Linear，数学上等价于一层 Linear，白搭。
        #   夹一个 SiLU 才有能力表达"离中心越远权重变化越快"这类弯的规律。
        # nn.Linear(64, 64)：再加工一轮，让这 64 维更好用。
        #
        # 补充一句：这里的坐标是被缩放过的"相对坐标"（大致落在 -1 ~ 1），
        # 不是原始像素值 0~511，所以数值不会太夸张。
        self.coord_embed = nn.Sequential(
            nn.Linear(2, coordinate_dim),
            nn.SiLU(),
            nn.Linear(coordinate_dim, coordinate_dim),
        )

        # ── 第三步：搭"街景照片整理器"（256 维 → 192 维 → 192 维）──
        # 输入是：在轮廓点所在位置，从整张图像特征图里"抠"出来的那一小块描述。
        # 这一层干两件活：
        #   ① 瘦身：256 → 192，腾出 64 个位置给门牌号；
        #   ② 翻译：把"通用图像理解"改写成"专门看轮廓"的语言。
        #      原始特征是为识别器官准备的，现在要它服务于找边界，
        #      所以需要一层重新加工。
        # 注意：这里是"一个点一个点独立处理"，点与点之间此刻互不通信。
        #       谁负责让点之间互相商量？是后面 DiT 里的自注意力。
        self.feat_embed = nn.Sequential(
            nn.Linear(feature_dim, feature_embedding_dim),
            nn.SiLU(),
            nn.Linear(feature_embedding_dim, feature_embedding_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        sampled_feat: torch.Tensor,
    ) -> torch.Tensor:
        """把两份资料整理好并装订成一张身份证。

        Args:
            x_t: 每个轮廓点的坐标。形状 [B, 点数, 2]。
                 B = 一次训练塞了几张图，点数是 128 个轮廓点，2 就是 (x, y)。
                 这个形状天然就是"一排点、每个点 2 个数字"，正合 Linear 的胃口。
            sampled_feat: 每个轮廓点位置上采到的图像特征。形状 [B, 通道数, 点数]。
                 注意顺序是"通道在前"，这是图像处理领域的老习惯
                 （长 × 宽 × 通道除外，卷积那边习惯通道放后面，但采样工具
                 吐出来常常是这个布局）。而 nn.Linear 只认"最后一维"，
                 所以要先把点数和通道数换个位置，见下面的 transpose。

        Returns:
            torch.Tensor: 每个点的完整身份证，形状 [B, 点数, 256]。
                 内容排布：前面 64 个数字 = 位置，后面 192 个数字 = 周围长相。
        """
        # 【处理门牌号】坐标已经是 [B, 点数, 2]，顺序正好，直接送进扩写器。
        # 出来变成 [B, 点数, 64]。
        coordinate_embedding = self.coord_embed(x_t)

        # 【处理街景照片】先调换顺序：[B, 通道, 点数] → [B, 点数, 通道]。
        # 为什么必须换？因为 nn.Linear 只会对"最后一维"做变换。
        # 不换的话，它会把"点数"当成特征去变换，那就完全搞错了对象。
        # 换完再送进整理器，出来变成 [B, 点数, 192]。
        # 这个 transpose 只是换个看法，不搬数据，几乎不花时间。
        feature_embedding = self.feat_embed(sampled_feat.transpose(1, 2))

        # 【装订】把两份资料在最后一维首尾相接，粘成一条 256 维的长条。
        # 结果：[B, 点数, 256]，每个点前 64 位说"我在哪"，后 192 位说"我周围啥样"。
        #
        # 为什么不在最开始就把 (x, y) 和特征拼起来，一起过一个大网络？
        #   因为两边的数值范围和含义差太远——坐标是小数字、特征是大数字，
        #   混在同一层里算，大的会把小的盖住。分开各走各的路，
        #   每边可以有自己的处理方式和宽度，最后再合并，才互不干扰。
        return torch.cat((coordinate_embedding, feature_embedding), dim=-1)


class MainlineDiTBlock(nn.Module):
    """【一个"轮廓调整工作室"】

    想象有 128 个工人，每人负责轮廓上的一个点。
    他们排成一个环，轮流（其实是同时）在这个工作室里干三件事：

        第一件（自注意力）—— 跟旁边的工人聊天：
            "你发现没有，我们都往外偏了，一起收一点吧？"
            这一步让 128 个点保持队形，不会各走各的、把轮廓撕断。

        第二件（交叉注意力）—— 抬头看墙上的医学图像：
            "我这块区域的骨头边界好像在这条线右边一点。"
            这一步把"眼睛看到的东西"注入进来，不然光靠聊天猜不出边界在哪。

        第三件（前馈网络 / MoE）—— 各自低头算一算，写下结论：
            "那我应该往右下挪 0.3 个像素。"

    干完这三件事，工作室就把新结论交出去，走进下一个工作室（下一个 DiT 块）。
    整个网络就是把这样的工作室串起来好几层，一波波地微调，轮廓越来越贴。

    这里面还有个"总导演"（adaLN 调制，见下面的 adaLN_modulation）：
    它根据"现在走到第几步（时间）"和"当前是什么阶段"，
    临时决定每个环节该听多少、该放大还是该压小。
    好比导演喊"这段戏收着演"，或者"这段戏放开演"。

    Args:
        dim: 每个点带多少维信息（默认 256，就是上一张"身份证"的厚度）。
        num_heads: 注意力拆成几组并行去看（默认 8，类似 8 个不同专业的顾问同看一份资料）。
        num_points: 轮廓上有多少个点（默认 128）。
        dropout: 训练时随机"捂住"一部分信息，防止死记硬背（默认 0 不用）。
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # 256 要能被 8 整除，否则均分给 8 个头会剩下零头，程序没法处理。
        # 这是一种"进门先检查"的写法：不满足就立刻报错，
        # 而不是等跑到一半才莫名其妙地崩掉。
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        # 256 ÷ 8 = 32：每个"顾问"分到 32 维信息。
        # 8 个顾问各自独立看一遍，最后把意见合起来，比一个人看全部更细。
        self.head_dim = self.dim // self.num_heads
        # 算分时用的一个缩放常数 1/√32。
        # 作用：把打分结果拉回正常范围，避免数值太大导致后面
        # 变成"只有一个人说话、其他全被忽略"的极端情况。
        self.scale = self.head_dim**-0.5

        # ── 三块"整理台"：每干一件事之前，先把数据摆整齐 ──
        # RMSNorm 做的事很朴素：把一组数字整体缩放到"平均大小正常"的水平。
        # 不做的话，数据经过一层层累积，有的数值越来越大、有的越来越小，
        # 最后训练会失控（要么爆掉、要么全变成 0）。
        # 三块对应三个环节，各管一段，互不干扰：
        #   norm1 → 自注意力前用；norm2 → 交叉注意力前用；norm3 → 前馈网络前用。
        self.norm1 = RMSNorm(self.dim)
        self.norm2 = RMSNorm(self.dim)
        self.norm3 = RMSNorm(self.dim)
        # 这道是专门给"提问"和"被查的资料"（Q 和 K）做的整理。
        # 为什么要单独管它俩？因为后面要拿它们做点积比较。
        # 如果一边数字特别大、另一边特别小，比较结果就会失真，
        # 注意力会变得"死盯一个点不放"或者"完全平均、毫无重点"。
        self.qk_norm = RMSNorm(self.head_dim)
        # 位置编码：告诉每个点"你在环上的第几个位置"。
        # 因为轮廓是个闭合圈，所以第 127 个点应该和第 0 个点算"邻居"。
        # 用"圆周角度"来表达位置，第 0 个是 0°，第 64 个是 180°，转一圈回来还是 0°。
        self.rope = CyclicRoPE1D(self.head_dim, num_points=num_points)

        # ── 自注意力的四个"翻译机"（都是 Linear，可以理解成可调的旋钮面板）──
        # 每个点带着 256 维信息进来，需要变成三个角色：
        #   Q（Query，我要问什么）
        #   K（Key，我这里有什么，可供别人来查）
        #   V（Value，如果别人选中我，我能提供什么内容）
        # 这三个角色都从同一份信息里现做，所以有三台翻译机。
        # 第四台 sa_out_proj 负责把讨论结果再整理回 256 维，交还给主流程。
        # bias=False：不加偏移量，省参数，大模型里这是常见做法。
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.sa_out_proj = nn.Linear(self.dim, self.dim, bias=False)

        # ── 交叉注意力的四台翻译机 ──
        # 这一组的对象变了：
        #   提问方（Query）仍然是我们这 128 个点，
        #   被查的资料（Key / Value）换成"墙上的医学图像"（image_context）。
        # 所以 Q 从点特征来，K 和 V 从图像特征来，两边翻译机要分开备。
        self.cross_q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.cross_k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.cross_v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.ca_out_proj = nn.Linear(self.dim, self.dim, bias=False)

        # ── "自己琢磨"的那台机器（前馈网络）──
        # 点们聊完、看完图像之后，每个点需要独立地再加工一遍自己的信息。
        # SwiGLU 是当前大模型里最常用的一种加工结构，效果比简单的两层线性更好。
        # 注意它并不看别的点，只看自己——"独立思考"阶段。
        self.mlp = SwiGLU(dim=self.dim, dropout=dropout)

        # ── 可选的"专家会诊"（MoE）──
        # 有些工作室（第 2/4/6 个）会额外养 4 位专科医生，
        # 遇到难处理的点就叫其中 2 位看诊，给一个补充意见。
        # 一开始这里是空的（None），由外面的代码决定要不要装上，见 enable_cr_moe()。
        self.routed_moe: nn.Module | None = None

        # ── 总导演（adaLN 调制）──
        # 输入是"当前进度条"（第几步 + 哪个阶段）的摘要，一共 256 个数。
        # 输出直接放大到 9 倍：9 × 256 个数，分成 9 摞，每摞 256 个。
        # 这 9 摞正好分给三个环节，每个环节拿 3 摞：
        #   一摞说"整体往负方向挪多少"（shift，平移）
        #   一摞说"每个通道放大多少倍"（scale，拉伸）
        #   一摞说"这段要不要演、演多大声"（gate，音量旋钮）
        # 先过 SiLU 只是个小弯折，让后面的线性层更有表达力。
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 9 * self.dim, bias=True),
        )
        # 关键的一手：把最后一层的旋钮全部归零。
        # 归零后，导演一开始只会喊"平移 0、放大 1 倍、音量 0"，
        # 等于什么都没改。这样网络刚出生时行为简单、好训练，
        # 后面再慢慢学会该怎么调节。（业界叫"零初始化"）
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # ── 一个"提前算好、反复使用"的小仓库（KV 缓存）──
        # 推理时同一个图像要反复用很多步（几十次小步调整）。
        # 图像部分每次都被重新翻译一遍太浪费，所以算一次存起来，后面直接取。
        # 训练时用不上（每次数据都不同），所以默认是空的。
        self._cached_k: Optional[torch.Tensor] = None
        self._cached_v: Optional[torch.Tensor] = None

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """【128 个点开一场圆桌会议，互相交流，保持队形】

        输入：每个点一份 256 维的资料，形状 [B, 128, 256]
        输出：每个点拿到一份"大家讨论后"的新资料，形状还是 [B, 128, 256]

        开会流程（就是标准注意力机制，一步一步看）：
          1. 每个点把自己的资料翻译成三个身份：我要问什么（Q）、我这里有什么（K）、
             我能提供什么（V）。
          2. 每个点的"问题"去和所有人的"资料标签"逐一比对，算出跟谁最相关。
          3. 按相关程度，把别人的"内容"按比例混进来。
          4. 合并 8 组顾问的意见，整理回 256 维。
        """
        batch_size, num_points, dim = x.shape
        # 第一步：翻译成 Q / K / V 三个身份。
        # 翻译完是 [B, 128, 256]，然后 .view(...) 把它切成 8 份，
        # 变成 [B, 128, 8, 32]，最后 .transpose(1, 2) 把"份数"提到前面，
        # 变成 [B, 8, 128, 32] —— 意思是：8 个顾问，每人手上 128 个点、每点 32 维。
        # 为什么要换位置？因为后面的矩阵运算要按"每个顾问独立处理"来跑。
        q = self.q_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # 第二步：先整理 Q 和 K 的数值范围（qk_norm），
        # 再给它们加上"我排在第几号位"的信息（apply_rotary）。
        # 加位置这一步很关键：如果只比内容不比位置，
        # 点 A 就分不清"跟我相邻的点"和"在对面的点"，
        # 队形会乱掉。加了圆周位置后，距离远近才被感知到。
        q = self.rope.apply_rotary(self.qk_norm(q))
        k = self.rope.apply_rotary(self.qk_norm(k))
        # 第三步：Q 和 K 逐个做点积，得到"相关度打分表"。
        # q @ k.transpose(-2, -1)：把 K 的最后两维掉个个儿才能相乘。
        # 结果是 [B, 8, 128, 128] —— 8 张 128×128 的打分表，
        # 第 i 行第 j 列 = "第 i 个点有多在意第 j 个点"。
        # 再乘 self.scale（约 0.177）把分数压回合理区间。
        attention = (q @ k.transpose(-2, -1)) * self.scale
        # softmax：把每一行的分数变成"百分比"，且加起来正好 100%。
        # 这一步让"重视谁、忽略谁"变成一个可以加权平均的分配方案。
        attention = attention.softmax(dim=-1)
        # 第四步：按百分比去取内容。attention @ v 就是"按权重把大家的内容混起来"。
        # 混完是 [B, 8, 128, 32]，把 8 份拼回 [B, 128, 256]：
        # transpose(1, 2) 把位置换回来 → contiguous() 让内存连续
        # （下面 view 要求内存连续，否则会报错）→ view 合并最后两维。
        output = (attention @ v).transpose(1, 2).contiguous().view(
            batch_size, num_points, dim
        )
        # 第五步：再过一台翻译机，把"会议纪要"整理成主流程能直接相加的格式。
        return self.sa_out_proj(output)

    def _cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """【128 个点拿着一张图查答案：我这儿到底该在哪】

        和上面的圆桌会议很像，但"被查资料"换人了：
          - 提问的仍然是 128 个点（x）
          - 被查的是医学图像的特征（context，可能是一大堆图像小块）

        打个比方：
          你是轮廓上的一个点，手里有个疑问"我这附近骨头边界在哪儿"。
          你拿着问题去翻图像这份大字典，找到跟你位置最相关的几页，
          把上面的信息抄回来。
        """
        batch_size, num_points, dim = x.shape
        # 只有提问方（轮廓点）需要现做。
        # 变成 8 个顾问各自提问的形状 [B, 8, 128, 32]。
        q = self.cross_q_proj(x).view(
            batch_size, num_points, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # 提问内容也先整理一下数值范围（这里不加位置编码，
        # 因为提问的是"空间里的点"，位置已由坐标本身携带）
        q = self.qk_norm(q)
        # 这里的 if/else 是个"省事开关"：
        #   如果之前已经把图像的 K、V 算好存起来了，直接拿来用（省时间）；
        #   否则现在现算一遍。
        # 为什么可以省？因为同一个图像在一轮推理里要查几十次，
        # 图像内容根本不变，翻译结果自然也不变。
        if self._cached_k is not None and self._cached_v is not None:
            k, v = self._cached_k, self._cached_v
        else:
            context_length = int(context.shape[1])
            k = self.cross_k_proj(context).view(
                batch_size, context_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v = self.cross_v_proj(context).view(
                batch_size, context_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k = self.qk_norm(k)
        # 打分、转百分比、按比例抄内容 —— 和圆桌会议完全一样的套路。
        # 只是这次打分表的形状是 [B, 8, 128, 图像块数量]：
        # 每一行是"我这个点，对图像每一块有多在意"。
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).contiguous().view(
            batch_size, num_points, dim
        )
        return self.ca_out_proj(output)

    def set_kv_cache(self, context: torch.Tensor) -> None:
        """【先把图像的"字典条目"抄好，后面反复查阅就快了】

        只做 K、V 这两份（不需要 Q，因为提问的是轮廓点，不是图像）。
        算完存进 self._cached_k / self._cached_v，下次 _cross_attention 直接用。
        注意 K 存的是整理过数值的版本，和正常路径保持一致，
        这样"开了缓存"和"没开缓存"算出来的结果完全相同。
        """
        batch_size, context_length, _ = context.shape
        k = self.cross_k_proj(context).view(
            batch_size, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.cross_v_proj(context).view(
            batch_size, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        self._cached_k = self.qk_norm(k)
        self._cached_v = v

    def clear_kv_cache(self) -> None:
        """【把抄好的条目扔掉】

        换了图像就必须清空，否则会拿着旧图的答案去回答新图的问题。
        """
        self._cached_k = None
        self._cached_v = None

    def forward(
        self,
        x: torch.Tensor,
        image_context: torch.Tensor,
        condition_embedding: torch.Tensor,
        route_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """【工作室的主流程：开会 → 看片 → 琢磨，三步走完加回去】

        Args:
            x: 128 个点的当前资料，形状 [B, 128, 256]。
                进门时它是"上一站的结果"，出门时变成"这一站调整后的结果"。
            image_context: 墙上挂的医学图像资料（可能是压缩过的全局概要，
                 也可能是某一小块局部细节，看这层是怎么配的）。
            condition_embedding: "进度条"摘要，[B, 256]。告诉总导演：
                 现在是第几步、处于哪个阶段，因此该收着改还是放开改。
            route_logits: 只有当这层装了"专家会诊"（MoE）时才需要传，
                 它说明每个点该找哪几位专家。普通层传 None。

        Returns:
            torch.Tensor: 调整后的 128 点资料，形状 [B, 128, 256]。
        """
        # ── 第 0 步：总导演先把"该怎么调"算出来 ──
        # 拿进度条查出一共 9 × 256 个数，然后 chunk(9, dim=1) 切成 9 摞，
        # 每摞 256 个，正好分给三个环节各 3 摞（平移 / 拉伸 / 音量）。
        # 名字里的 sa = self-attention（自注意力）、ca = cross-attention（交叉注意力）、
        # ff = feed-forward（前馈网络）。
        modulation = self.adaLN_modulation(condition_embedding)
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_ff,
            scale_ff,
            gate_ff,
        ) = modulation.chunk(9, dim=1)

        # ── 第 1 步：圆桌会议（点跟点聊）──
        # 顺序是：先 norm1 把数据摆整齐 → modulate 按导演要求平移+拉伸
        # → 再去开会 → 结果乘以"音量旋钮"gate_sa → 加回原来的 x。
        # 最后这个"加回原来的 x"叫残差连接，非常重要：
        #   它保证这一层最多是"在原有基础上微调"，不会把之前辛苦积累的
        #   信息一把抹掉。就算这一步学得很差，x 本身还在，网络不至于崩。
        # gate_sa.unsqueeze(1)：gate 形状是 [B, 256]，而 x 是 [B, 128, 256]，
        #   点这一维得补上，才能让每个点都用同一套旋钮。
        self_attention_input = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(self_attention_input)

        # ── 第 2 步：看片子（点去查图像）──
        # 结构和上一步一样，只是"查的对象"从别的点换成了图像。
        # 注意传进去的 x 是上一步已经更新过的版本——信息是一路叠加的。
        cross_attention_input = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(
            cross_attention_input,
            image_context,
        )

        # ── 第 3 步：各自低头琢磨 ──
        # 先做基础加工（self.mlp，每个点独立处理，不跟别人交流）。
        ffn_input = modulate(self.norm3(x), shift_ff, scale_ff)
        ffn_output = self.mlp(ffn_input)
        # 如果这层装了专家会诊，就再请专家给一份"补充意见"，
        # 直接加到基础结果上（叫"残差专家"：专家只负责补差，不推翻重来）。
        #
        # 下面两个报错是"配对检查"，防止配置搞错：
        #   装了专家却没给 route_logits → 专家不知道该怎么分配，报错；
        #   没装专家却传了 route_logits → 说明外面以为这层有专家，报错。
        # 这类检查的意义是：宁可立刻崩掉并说明原因，也不要悄悄算错。
        if self.routed_moe is not None:
            if route_logits is None:
                raise RuntimeError("CR-MoE route logits are required")
            ffn_output = ffn_output + self.routed_moe(ffn_input, route_logits)
        elif route_logits is not None:
            raise RuntimeError("route logits supplied to a dense DiT block")

        # ── 第 4 步：琢磨的结果也按音量旋钮调一下，再加回 x ──
        # 同样是残差连接。至此这一站的工作结束，x 交给下一个工作室。
        return x + gate_ff.unsqueeze(1) * ffn_output

    def enable_cr_moe(self) -> None:
        """【给这个工作室请 4 位专科医生坐诊】

        不是每个工作室都需要——只在中间几层（第 2/4/6 个）请，
        因为那些层处理的是"难啃的细节"，值得多花点算力。

        4 位医生，每个点只看其中 2 位（Top-2 路由）：
            - 为什么不 4 位全看？太贵。而且全看等于又变回一个普通的大网络，
              "分专科"就没意义了。
            - 为什么不全看 1 位？那样每个点只有一种视角，容易钻牛角尖。
        每位医生内部有 256 维的思考空间（expert_hidden_dim=256），
        和主流程一样宽，够用。

        后面那三个权重是给训练时的"纪律约束"用的：
            load_balance_weight：防止 4 位医生里某一位被点爆、其余闲着。
                理想情况是 4 位都接到差不多数量的病人。
            specialization_weight：逼医生们去看不同的病。否则 4 个人
                都成了全科大夫，等于白请。
            expert_init_std：医生初始"水平"要非常接近零（1e-4）。
                让专家一开始几乎不发声，需要时再慢慢学会说话，
                这样不会一上来就把主流程搅乱。
        """
        # 已经请过了就别再请一遍，否则会把学好的专家直接覆盖掉。
        if self.routed_moe is not None:
            raise RuntimeError("this DiT block already has routed experts")
        # 这里才 import，是为了避免循环引用（cr_moe 那边可能也 import 本文件的东西）。
        from .cr_moe import ContourSharedResidualExperts

        self.routed_moe = ContourSharedResidualExperts(
            dim=self.dim,
            num_experts=4,
            top_k=2,
            expert_hidden_dim=256,
            load_balance_weight=1e-2,
            specialization_weight=5e-3,
            expert_init_std=1e-4,
        )

    def reg_loss(self) -> torch.Tensor:
        """【领一张给医生的"纪律罚单"】

        训练时除了主任务损失，还要额外加上这个纪律损失，
        逼 4 位医生"活儿分匀、各看各的病"。
        没有专家的工作室罚单金额为 0，不影响总损失。
        """
        if self.routed_moe is None:
            # 借一个现成的参数张量造个 0，好处是它自动落在正确的设备/精度上
            # （参数在 GPU 上就造 GPU 上的 0），也不多占内存。
            return self.adaLN_modulation[-1].weight.new_zeros(())
        return self.routed_moe.reg_loss()


class DenseResidualFinalHead(nn.Module):
    """【最后一道工序：把厚厚的"想法"压成两个数字】

    走到这一步，128 个点每人手上都攒着一份 256 维的"想法"——
    是他们开完会、看完片子、各自琢磨之后攒下来的。
    但系统真正需要的答案极其简单，每个点只要两个数字：

        我该往右挪多少？（dx）
        我该往下挪多少？（dy）

    所以这一层干的活就是**翻译**：256 个数 → 2 个数。

    翻译工作请了两位师傅同时干，最后把两份答案加在一起：

        师傅甲（linear）：直来直去，像一把尺子，判断"整体该往哪个方向挪"。
                          他擅长把握大方向，但只会画直线。

        师傅乙（residual_mlp）：中间绕到 1024 维再折回来，专门抓小细节。
                                比如骨头边缘有个小凹口、或者局部有点弯，
                                尺子量不出来，他负责补上这一点点。

    为什么非要两个人？只用师傅甲，轮廓会被画成一圈硬邦邦的直线段；
    只用师傅乙，容易学歪、训练不稳。一个管大方向、一个补小细节，配合最靠谱。
    这也是名字里 "Dense + Residual" 的来历：主干 + 残差补丁。

    还有一位"场务"（adaLN）：在翻译之前先把这 256 个数整理一下，
    告诉它"现在走到这个阶段了，语气该收着还是放开"。

    最终输出形状 [B, 128, 2]：每张图、每个点、两个数字（dx, dy）。

    Args:
        dim: 输入"想法"的厚度（默认 256）。
        out_dim: 要输出几个数字（固定 2，就是 x 和 y 的位移）。
        hidden_dim: 师傅乙中途绕路时的宽度（默认 1024，比 256 宽很多，
                    因为"抓细节"需要在更宽的空间里折腾）。
        residual_init_std: 师傅乙最后一层的初始"音量"（1e-4，小到几乎听不见，
                    原因见下面初始化部分）。
    """

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        hidden_dim: int = 1024,
        residual_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        # 翻译前先整理一下数据：把 256 个数的整体大小调到正常范围。
        # 不整理的话，前面几层累积下来的数值可能忽大忽小，翻译会失准。
        self.norm = RMSNorm(dim)
        # 师傅甲：一把直尺，256 个数直接量出 2 个数。
        self.linear = nn.Linear(dim, out_dim)
        # 场务：拿"进度条"（第几步、什么阶段）算出两组旋钮 —— 平移量和平滑量。
        # 输出 2 * dim 个数，正好切成两半，一半管"整体挪一挪"，
        # 一半管"整体松一松/紧一紧"。名字叫 adaLN，就是"按条件调一调"的意思。
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        # 师傅乙：先撑到 1024 维（有地方折腾），中间弯一下（SiLU），再压回 2 个数。
        # 这个"撑开—弯曲—压回"的形状，就是他比直尺更能抓细节的原因。
        self.residual_mlp = nn.Sequential(
            nn.Linear(dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), out_dim),
        )
        # ── 初始化：让网络"出生时原地不动" ──
        # 这一段是全篇最讲究的地方，逐条说：
        #
        # ① 场务的旋钮全部归零：
        #    旋钮为 0 时，整理动作变成"什么都不改"（平移 0、缩放 1 倍）。
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        #
        # ② 师傅甲的旋钮全部归零：
        #    这样他一开始输出的两个数字**严格等于 0**，
        #    等于在说"谁都别动"。
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        #
        # ③ 师傅乙第一层用常规的随机初始化（xavier）：
        #    他需要一个正常工作的"入口"，否则什么都学不到。
        #    偏置设成 0 只是去掉多余的起点偏移。
        nn.init.xavier_uniform_(self.residual_mlp[0].weight)
        nn.init.zeros_(self.residual_mlp[0].bias)
        #
        # ④ 师傅乙最后一层用极小的随机数（标准差 1e-4）：
        #    他内部照常在算，但"说出口的音量"被压到几乎为零，
        #    输出大约只有万分之几，肉眼看着就像 0。
        #
        # 前四条合起来的效果：网络刚出生时，预测的位移 ≈ 0，
        # 也就是说轮廓**先站着不动**，从一个"最保险的答案"开始学。
        # 这比一上来就乱动要好训得多 —— 乱动的起点往往意味着巨大的初始误差。
        nn.init.normal_(self.residual_mlp[-1].weight, std=float(residual_init_std))
        nn.init.normal_(self.residual_mlp[-1].bias, std=float(residual_init_std))

    def forward(
        self,
        x: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """【翻译：256 个数 → 两个数字（dx, dy）】

        Args:
            x: 最后一层 DiT 交出来的"想法"，形状 [B, 128, 256]。
            condition_embedding: "进度条"摘要，[B, 256]。
                 告诉场务：现在第几步、什么阶段，因此该收着还是放开。

        Returns:
            torch.Tensor: 每个点该挪多少，形状 [B, 128, 2]。
                 最后那一维就是 (dx, dy)。数值小表示"几乎别动"，
                 数值大表示"这个点要明显调整"。
        """
        # 场务出场：算出两组旋钮，然后把 2*256 个数从中间切成两半。
        # 前半 = 平移量 shift，后半 = 缩放量 scale，各 256 个。
        shift, scale = self.adaLN(condition_embedding).chunk(2, dim=1)
        # 按旋钮整理数据：先归一化，再"整体挪一挪 + 松一松"。
        # 注意 shift / scale 形状是 [B, 256]，而 x 是 [B, 128, 256]，
        # 广播机制会自动让同一张图的 128 个点共用同一套旋钮。
        x = modulate(self.norm(x), shift, scale)
        # 两位师傅同时开工，答案相加。
        # 注意他们吃的是**同一份**整理后的 x，只是各自的看法不同。
        # 相加而不是接续，是为了让"大方向"和"小细节"平级叠加，谁也不压谁。
        return self.linear(x) + self.residual_mlp(x)

    def reg_loss(self) -> torch.Tensor:
        """【交一张空白罚单】

        别的层有专家时要交"纪律罚单"（负载均衡、专家分化）。
        这一层没有专家，所以罚单金额固定为 0。
        之所以还要保留这个方法，是为了让外面收集损失时可以**无脑统一调用**，
        不用挨个判断"这层到底有没有专家"。返回 0 就等于"没意见"。
        """
        # 借 self.linear 的权重造个 0：它会自动落在同一块设备、同一种精度上，
        # 也不需要额外申请内存。
        return self.linear.weight.new_zeros(())


class MainlineFlowDenoiser(nn.Module):
    """【整台机器：前台备料 + 6 个工作室串成流水线 + 出口打包】

    前面讲的 MainlineDiTBlock 只是"一个轮廓调整工作室"。
    这个类负责把整条流水线搭起来：进门前先备料，中间过 6 个工作室，
    最后在出口把结果压成"每个点该往哪挪"。

    ─────────────── 进门：备料区 ───────────────

        time_emb_net       把"现在走到第几步"翻译成一串数字
        s_emb_net          把"大阶段进度"翻译成一串数字
                           （两者相加 = 一张"进度条摘要"）
        global_compressor  把整张片子压成一小叠"全局摘要卡"
                           （细节全丢了，但整体形状和位置关系还在）
        local_proj         把每个点位置采到的局部细节整理干净
        point_embed        给每个点做一张"身份证"
                           （前 64 位写"我在哪"，后 192 位写"我周围啥样"）

    ─────────────── 中间：6 个工作室 ───────────────

    编号 0 到 5，依次通过。中间有个关键设计：**交替看两种材料**。

        工作室 0（偶数）→ 抬头看"全局摘要卡"：整体该是这么个形状
        工作室 1（奇数）→ 低头看"局部细节"  ：这一小段的边到底在哪
        工作室 2（偶数）→ 又看全局
        工作室 3（奇数）→ 又看局部
        ...

    为什么这么交替？就像画画：
        先起大形（全局）→ 凑近抠边角（局部）→ 退后看一眼整体有没有被抠歪
        （全局）→ 再凑近修一遍（局部）。
    只盯全局，边缘永远糊；只盯局部，容易把整体形状带跑偏。交替才稳。

    ─────────────── 调度员：路由器 ───────────────

        ContourRoutePath 一次性决定：这张片子在第 2/4/6 个工作室里
        该找哪 2 位专科医生看诊。三个工作室共用一次决策，不重复算。

    ─────────────── 出口：打包 ───────────────

        DenseResidualFinalHead 把 256 维想法压成 (dx, dy) 两个数字。

    ─────────────── 记住这条数据流，全篇就通了 ───────────────

        进门：坐标 [N, 128, 2]  +  图像特征 [B, C, H, W]
        途中：每个点被撑成 [N, 128, 256]
        出门：[N, 128, 2]（每个点两个数字）

    小心这里有两个不同的"批"：
        N = 轮廓条数（一张片子上可能有左右两条轮廓，所以 N ≥ B）
        B = 图像张数
    两者不相等时，靠 py_ind 这个"对照表"把轮廓和它所属的图对上号。

    Args:
        state_dim: 每个点"想法"的厚度（默认 256）。
        feature_dim: 图像特征原本多少个通道（默认 256）。
        num_layers: 工作室个数。这里**强制必须写 6**，写别的直接报错。
        num_heads: 每个工作室里注意力拆成几组并行看（默认 8）。
        num_points: 轮廓点数（默认 128）。
        num_queries: 全局摘要卡要做几张（默认 256）。
        dense_residual_hidden_dim: 出口那位"抓细节"的师傅中途绕多宽（默认 1024）。
        use_s_conditioning: 是否使用"阶段进度"这个条件。必须为 True——
            整条流程靠它区分"粗调"和"精修"，不给就没法工作。
    """

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 128,
        num_queries: int = 256,
        dense_residual_hidden_dim: int = 1024,
        use_s_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.time_dim = self.state_dim
        self.num_points = int(num_points)
        # 为什么死死卡住 6 层？因为路由器的输出、专家层的挂载位置（1/3/5）
        # 都是按 6 层写死的。允许改成 4 层或 8 层，这些位置就全错位了，
        # 而且不会报错、只会悄悄算错——所以干脆在这里一进门就拦住。
        if int(num_layers) != 6:
            raise ValueError("CR-MoE mainline requires exactly six DiT blocks")

        # ── 备料 ①：把"现在第几步"翻译成 256 个数字 ──
        # 先由 SinusoidalTimeEmbedding 把 1 个数字 t 变成 64 个波（不同频率的正弦），
        # 再用两层 Linear 加工到 256 维。
        # 为什么要先变成"波"？因为单纯的数字 0.37 太单薄，网络不好用；
        # 变成一组不同频率的波之后，相近的 t 会有相似的波形，
        # 网络就能"感觉到"步与步之间的远近关系。
        # 这里 64 → 256 是升维：给网络更多空间去表达"这个时刻意味着什么"。
        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=self.state_dim // 4),
            nn.Linear(self.state_dim // 4, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        # ── 备料 ②：把整张片子压成"全局摘要卡" ──
        # 输入是 64×64 的图像特征图（4096 个小块），输出是 256 张摘要卡。
        # 条目数从 4096 砍到 256，砍掉了大量细节，只留下"整体形状、位置、
        # 大致解剖结构"这类宏观信息。这就是偶数层看到的材料。
        self.global_compressor = PerceiverCompressor(
            in_dim=self.feature_dim,
            out_dim=self.state_dim,
            num_queries=num_queries,
        )
        # ── 备料 ③：整理"局部细节" ──
        # 每个轮廓点在自己的位置上，从特征图上抠出一小条 256 维的"周围长啥样"。
        # 这里只是把它从 256 通道整理成 256 维的标准格式，顺便换个"专用视角"。
        # 奇数层看到的就是这份材料——全是边缘、纹理这类高频细节。
        self.local_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        # Preserve the effective YOLO RAOS model's normal/tangent boundary cues.
        # Six finite-difference feature terms, each with feature_dim channels.
        self.detail_local_proj = nn.Sequential(
            nn.Linear(6 * self.feature_dim, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.detail_point_proj = nn.Sequential(
            nn.Linear(6 * self.feature_dim, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        for projection in (self.detail_local_proj, self.detail_point_proj):
            nn.init.zeros_(projection[-1].weight)
            nn.init.zeros_(projection[-1].bias)
        # ── 备料 ④：给每个点做身份证 ──
        # 把"在哪儿"（坐标）和"周围长啥样"（局部特征）分别整理后拼起来，
        # 得到每点 256 维的完整描述。这是 6 个工作室的**输入**。
        self.point_embed = SeparatePointEmbedding(
            state_dim=self.state_dim,
            feature_dim=self.feature_dim,
        )
        # ── 搭 6 个工作室，串成一排 ──
        # ModuleList 就是一个"带登记的列表"，PyTorch 会自动管理里面所有参数。
        # dropout=0.0：这个模型不用 dropout（数据量够、而且扩散模型本身有正则作用）。
        self.dit_layers = nn.ModuleList(
            [
                MainlineDiTBlock(
                    dim=self.state_dim,
                    num_heads=num_heads,
                    num_points=self.num_points,
                    dropout=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        from .cr_moe import ContourRoutePath

        # ── 决定哪几个工作室请专科医生 ──
        # 注意这里的坑：代码里写的是 Python 索引，从 0 开始数。
        #   Python 索引 1/3/5  =  人眼数的第 2/4/6 个
        # 只请中间三层的理由：浅层还在整理基础信息，深层已经在做最后收尾，
        # 中间这几层才是"啃硬骨头"的地方，加专家最划算。
        self._cr_moe_layer_indices = (1, 3, 5)
        # ── 请一位调度员，管三个工作室的分诊 ──
        # num_routed_blocks=3 表示它一次输出三组决策（对应三个专家层），
        # 而不是每层各配一个调度员。这样做的好处：
        #   ① 只算一次，省算力；
        #   ② 三个层的决策出自同一个"大脑"，彼此有关联，不会各说各话。
        self._contour_moe_router = ContourRoutePath(
            dim=self.state_dim,
            num_routed_blocks=len(self._cr_moe_layer_indices),
            num_experts=4,
            hidden_dim=256,
            temperature=1.0,
        )
        # 挨个给这三个工作室"装上"专家。
        # 注意：装的动作发生在**建完所有层之后**，而不是在构造函数参数里，
        # 这样其余三个层就天然没有专家，不浪费任何参数。
        for index in self._cr_moe_layer_indices:
            self.dit_layers[index].enable_cr_moe()

        # ── 备料 ⑤：把"大阶段进度"也翻译成 256 个数字 ──
        # 和 time_emb_net 结构一模一样，但含义不同：
        #   t（时间步）说的是"扩散流程走到多细了"，
        #   s（阶段）说的是"外层演化走到第几轮了"。
        # 两者是两条独立的时间轴，所以要各配一套翻译网络。
        #
        # 为什么必须留着 s？因为整条流程分"粗调 / 精修"两种模式，
        # 同一个输入在不同阶段该做出不同幅度的调整，全靠这个信号区分。
        if not use_s_conditioning:
            raise ValueError("the mainline requires diffusion-stage conditioning")
        self.s_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=self.state_dim // 4),
            nn.Linear(self.state_dim // 4, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        # 最后这层归零 = 出生时"阶段嵌入贡献为 0"。
        # 效果是：训练刚开始时，模型只按"时间步 t"办事，先学会基本盘；
        # 之后 s 这个维度再慢慢长出作用，学会"分阶段调整力度"。
        # 这是一种很常见的"渐进式上线"手法，避免两个新东西同时乱动。
        nn.init.zeros_(self.s_emb_net[-1].weight)
        nn.init.zeros_(self.s_emb_net[-1].bias)

        # ── 出口：把 256 维想法压成 (dx, dy) ──
        # 细节见 DenseResidualFinalHead 的注释，这里只说一句：
        # 它内部也做了零初始化，所以整个网络"出生时预测位移为 0"。
        self.final_layer = DenseResidualFinalHead(
            dim=self.state_dim,
            out_dim=2,
            hidden_dim=dense_residual_hidden_dim,
            residual_init_std=1e-4,
        )

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        py_ind: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        detail_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """【整场演出：备料 → 6 站 → 打包】

        Args:
            cnn_feature: 整张片子的图像特征 [B, C, H, W]。
                 B 是图像张数，H/W 是 64×64。相当于"墙上的片子"。
            sampled_feat: 每个轮廓点位置上抠出的局部特征 [N, C, 128]。
                 注意顺序是"通道在前"，因为这是采样工具吐出来的格式。
            x_t: 当前这 128 个点在哪 [N, 128, 2]。
                 注意这是**当前时刻的轮廓**，也就是"待打磨的粗坯"。
            t: 扩散时间步，每个轮廓一个数 [N]。
                 可以理解成"打磨进度条"：0 附近是粗坯，1 附近是成品。
            py_ind: 对照表 [N]。每个轮廓登记"我来自第几张片子"，
                 比如 [0, 0, 1] 表示前两条轮廓都属于第 0 张图。
                 不传的话就必须靠数量猜，详见下面第 2 步。
            s: 外层大阶段进度 [N]，0.0 = 第一次粗更新，1.0 = 精细调整。

        Returns:
            两个东西：
                - prediction [N, 128, 2]：每个点该往哪挪（速度场）
                - regularization：一个标量。三个专家层交的"纪律罚单"加起来。
                  它不监督"预测得准不准"，只约束"专家们要分工均匀"，属于
                  训练时附加的辅助项。它会正常回传梯度（专门去调 router 和专家），
                  但推理时没有意义，直接忽略即可。
        """
        # ── 进门先核对证件 ──
        # 这四行是"丑话先说在前头"：与其让错误数据一路混进去、
        # 最后在某个莫名其妙的地方报出看不懂的错，不如在这里就拦住，
        # 并且把"我期望什么形状、你给我什么形状"一起打出来。
        if x_t.ndim != 3 or x_t.shape[-1] != 2:
            raise ValueError(f"expected x_t [N,P,2], got {tuple(x_t.shape)}")
        if t.ndim != 1:
            raise ValueError(f"expected t [N], got {tuple(t.shape)}")
        if sampled_feat.ndim != 3:
            raise ValueError(
                f"expected sampled_feat [N,C,P], got {tuple(sampled_feat.shape)}"
            )
        if s is None:
            raise ValueError("the mainline denoiser requires stage progress s")

        # ── 统一精度 ──
        # 模型参数可能是 fp32、也可能是 fp16/bf16。如果输入是 fp32 而模型是 fp16，
        # 两者一相乘就会报错（"dtype mismatch"）。
        # 这里读一下模型当前用的精度，然后让所有输入都跟上，省得后面到处出错。
        parameter_dtype = next(self.parameters()).dtype
        cnn_feature = cnn_feature.to(parameter_dtype)
        sampled_feat = sampled_feat.to(parameter_dtype)
        x_t = x_t.to(parameter_dtype)
        t = t.to(parameter_dtype)
        s = s.to(parameter_dtype)

        contour_count = int(x_t.shape[0])
        # ═══ 第 1 步：把两条时间轴都翻译成"进度条摘要" ═══
        # 先各自翻译，再相加。为什么是相加而不是拼接？
        # 相加相当于"两个提示叠在一起"，网络自己去学怎么权衡；
        # 拼接会把维度翻倍，还得多加一层去融合。相加更省事，实践中也够用。
        time_embedding = self.time_emb_net(t)
        # 这里 s * 1000 是个小技巧：s 原本落在 0~1 之间，直接喂给"正弦波编码"
        # 的话，不同 s 生成的波形几乎一样，区分不开（就像用毫米刻度尺量纳米）。
        # 乘以 1000 把范围拉到 0~1000，波形差异才明显。
        stage_embedding = self.s_emb_net(
            (s * 1000.0).to(device=t.device, dtype=time_embedding.dtype)
        )
        condition_embedding = time_embedding + stage_embedding

        # ═══ 第 2 步：压全局摘要，并把轮廓和它所属的片子对上号 ═══
        # 万一上游只传了一张没带 batch 维的特征图，这里补一个"虚拟批"。
        if cnn_feature.ndim == 3:
            cnn_feature = cnn_feature.unsqueeze(0)
        # 4096 个小块 → 256 张摘要卡。此时形状 [B, 256, 256]。
        global_context = self.global_compressor(cnn_feature)
        # 关键：摘要卡是按"图像"算出来的，但下游是按"轮廓"排队的。
        # 一张图上有两条轮廓时，两边数量就对不上了。三种情况分别处理：
        if py_ind is not None:
            # ① 有对照表 → 照着表取。第 i 条轮廓取第 py_ind[i] 张图的摘要卡。
            #    这一步会产生新的张量（复制），但只有 256 条卡片，很便宜。
            global_context = global_context[py_ind]
        elif global_context.shape[0] != contour_count:
            if global_context.shape[0] == 1:
                # ② 没对照表，但只有一张图 → 让所有轮廓共用这张图的摘要。
                #    expand 只改"视图"，不真的复制内存，所以很省。
                global_context = global_context.expand(contour_count, -1, -1)
            else:
                # ③ 没对照表、图又不止一张、数量还对不上 → 无法判断谁配谁，
                #    直接报错。绝不"猜一个"然后默默算错。
                raise ValueError(
                    "image/contour batch mismatch: "
                    f"global={global_context.shape[0]}, contours={contour_count}"
                )

        # ═══ 第 3 步：备好局部材料，并给每个点做身份证 ═══
        # sampled_feat 是 [N, C, 128]（通道在前），先换成 [N, 128, C]
        # 才能让 Linear 作用在通道上。整理完就是奇数层要看的"局部细节"。
        local_context = self.local_proj(sampled_feat.transpose(1, 2))
        # 身份证：坐标 64 维 + 局部特征 192 维 = 256 维。
        # 注意这里直接把原始 sampled_feat（没经过 local_proj）喂进去——
        # 两条支路各取所需，相互独立，不是串联关系。
        x = self.point_embed(x_t, sampled_feat)
        if detail_feat is None:
            raise ValueError("RAOS YOLO evolution requires normal/tangent detail features")
        detail_context = detail_feat.transpose(1, 2).to(parameter_dtype)
        local_context = local_context + self.detail_local_proj(detail_context)
        x = x + self.detail_point_proj(detail_context)

        # ═══ 第 4 步：调度员一次性定好三层的分诊方案 ═══
        # 看的是 local_context（局部细节），输出形状 [N, 3, 4]：
        #   N 条轮廓 × 3 个专家层 × 4 位专家各打一个分。
        route_path = self._contour_moe_router(local_context)
        # 建一张"地址簿"：层索引 → 在 route_path 里排第几。
        # 结果是 {1: 0, 3: 1, 5: 2}，意思是"第 1 层的决策存在位置 0"。
        # 有了它，下面循环里就能一句话取出对应那一组决策。
        route_positions = {
            layer_index: route_index
            for route_index, layer_index in enumerate(self._cr_moe_layer_indices)
        }

        # ═══ 第 5 步：6 站流水线，一站站过 ═══
        for index, layer in enumerate(self.dit_layers):
            # 交替换材料：偶数站看全局摘要（定大形），奇数站看局部细节（抠边角）。
            context = global_context if index % 2 == 0 else local_context
            # 只有 1/3/5 这三站有专家，才需要递分诊单；其余站给 None。
            # route_path[:, k, :] 的意思是：所有轮廓、第 k 组决策、4 个分数。
            # 结果形状 [N, 4]，正好是"每条轮廓对 4 位医生的打分"。
            route_logits = (
                route_path[:, route_positions[index], :]
                if index in route_positions
                else None
            )
            # 一站处理完，x 被更新，接着送进下一站。
            # 注意 x 的形状 [N, 128, 256] 全程不变，变的只是内容。
            x = layer(
                x,
                context,
                condition_embedding,
                route_logits=route_logits,
            )

        # ═══ 第 6 步：打包出门 ═══
        # 256 维想法 → 2 个数字（dx, dy）。
        prediction = self.final_layer(x, condition_embedding)
        # 把三个专家层交的"纪律罚单"加起来。
        # 先从出口层起头（它固定交 0），这样后面可以做纯粹的加法，
        # 不用纠结"第一个该赋什么初始值"。
        regularization = self.final_layer.reg_loss()
        for layer in self.dit_layers:
            regularization = regularization + layer.reg_loss()
        # 没有专家的层会交 0，所以这个循环对所有层都安全。
        return prediction, regularization

    def cr_moe_diagnostics(self) -> dict[str, torch.Tensor]:
        """【体检报告：专家们到底有没有在干活】

        返回几个数字，用来判断 MoE 是不是白请了：
            route_entropy       调度员做决定时有多"果断"。接近 0 说明总选同一位，
                                等于没分诊；太高说明糊里糊涂随机选。
            dead_experts        从头到尾没接到病人的医生数量。大于 0 就是浪费。
            soft_load_min       接诊量最少的医生的忙碌程度。
            adjacent_block_agreement  三个专家层的决策有多一致。

        训练时盯着这几个数，就能判断该不该继续留着这套专家结构。
        这些数字都做了 detach，只是用来"看"，不会参与反向传播。
        """
        from .cr_moe import collect_cr_moe_diagnostics

        return collect_cr_moe_diagnostics(self)

    # 起个别名，让调用方既能写 cr_moe_diagnostics() 也能写 moe_diagnostics()，
    # 两种写法指向同一个函数，纯属方便。
    moe_diagnostics = cr_moe_diagnostics


# 这个文件对外提供三样东西，别的文件 from ... import 时只能拿到这三个：
#   输出头、单个工作室、整台机器（按由小到大的顺序排列）
__all__ = (
    "DenseResidualFinalHead",
    "MainlineDiTBlock",
    "MainlineFlowDenoiser",
)
