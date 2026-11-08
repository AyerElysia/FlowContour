"""两个工具函数，分别服务于两个训练阶段：

    mixed_states   → 预训练 pretrain：随机挑"起点轮廓"，让网络见过各种情况
    terminal_credit → GDPO：把终点分数拆成两步各自的功劳，用来更新策略

监督预训练沿用已验证 RAOS 版本的连续外层进度采样；GDPO 使用组相对终点回报。
"""

import torch
from lib.config import cfg


# ═══════════════════════════════════════════════════════════════
# 函数一：预训练时随机挑"起点轮廓"
# ═══════════════════════════════════════════════════════════════
def mixed_states(initial, target, probabilities=None, refine_sigma_px=None, refine_modes=None):
    """复用旧版连续课程：current = initial + s * (GT - initial)。

    s 是实际完成进度，不是随机粗调/精修标签；目标为 GT-current 的剩余位移。
    从五中心高斯混合中采样进度，并以 15% 概率用均匀背景覆盖。
    最后三个参数只保留旧调用接口；课程由 cfg.v4_10_* 明确定义。
    kind 按进度分成三箱，仅供日志统计，不参与网络条件。
    """
    count = initial.shape[0]
    centers = initial.new_tensor(cfg.v4_10_centers)
    weights = initial.new_tensor(cfg.v4_10_center_weights)
    if centers.ndim != 1 or centers.numel() != weights.numel() or centers.numel() == 0:
        raise ValueError("continuous-state centers and weights must be equal nonempty vectors")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("continuous-state weights must be nonnegative with positive sum")
    centers = centers.clamp(0.0, 0.999)
    weights = weights / weights.sum()
    draw = torch.rand(count, device=initial.device, dtype=initial.dtype)
    center_index = (draw[:, None] >= weights.cumsum(0)[None]).sum(1).clamp_max(centers.numel() - 1)
    gaussian = (centers[center_index] + torch.randn_like(draw) * float(cfg.v4_10_sigma)).clamp(0.0, 0.999)
    background = torch.rand(count, device=initial.device) < float(cfg.v4_10_uniform_bg_prob)
    uniform = torch.rand(count, device=initial.device, dtype=initial.dtype) * 0.999
    phase = torch.where(background, uniform, gaussian)
    current = initial + phase[:, None, None] * (target - initial)
    # Bins are log diagnostics only; phase remains the actual cumulative progress.
    kind = (phase >= 0.2).long() + (phase >= 0.8).long()
    return current, phase, kind


# ═══════════════════════════════════════════════════════════════
# 函数二：GDPO 的终点信用分配
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def terminal_credit(rewards, stage_quality, std_floor=.1, clip=2.):
    """把"整条轨迹的终点分数"拆成两步各自的功劳，用于 PPO 更新。

    背景：GDPO 跑 8 条候选轨迹，每条走两步（粗调→精修）。
    最后只在终点算一个 NSD 分数。但更新时要分别给两步的动作打分，
    所以需要把终点分数"分摊"到两步上。

    分摊原则：哪一步结束时的绝对质量更高，就把更多功劳归给哪一步。
    （不是各步独立归一化，而是按绝对质量比例分配。）

    参数：
        rewards       ：[K, N]，K 条候选轨迹、N 条轮廓的终点 NSD 分数
        stage_quality ：[K, 2, N]，每步结束时的绝对质量（0~1）
        std_floor     ：标准差下限，防止某条轮廓 8 条候选分数全一样导致除零
        clip          ：优势值的裁剪范围（±2），防止极端分数带偏梯度

    返回：
        advantage ：[K, N]，每条轨迹的"标准化优势"（好轨迹正、差轨迹负）
        weights   ：[K, 2, N]，两步各自应该分到的比例（加起来 = 2）
        allocated ：[K, 2, N]，两步各自的实际优势（= advantage × weights）
    """
    # 参数合法性检查
    if std_floor <= 0 or clip <= 0:
        raise ValueError("advantage floor and clip must be positive")
    if rewards.ndim != 2 or stage_quality.shape != (rewards.shape[0], 2, rewards.shape[1]):
        raise ValueError("expected terminal [K,N], stage quality [K,2,N]")
    if not torch.isfinite(stage_quality).all() or not torch.isfinite(rewards).all():
        raise ValueError("nonfinite quality")
    if (stage_quality < 0).any() or (stage_quality > 1).any():
        raise ValueError("stage quality must be absolute quality in [0,1]")

    # ① 算优势：每条轨迹的分数 − 同类轨迹的平均分，再除以标准差
    # 这样好轨迹得到正优势，差轨迹得到负优势
    # clamp_min(std_floor)：如果 8 条轨迹分数几乎一样（标准差≈0），
    # 就用 std_floor 兜底，避免除以极小值导致优势爆炸
    # clamp(-clip, clip)：把优势限制在 ±2 之间，防止极端值带飞梯度
    advantage = ((rewards - rewards.mean(0)) /
                 rewards.std(0, unbiased=False).clamp_min(std_floor)).clamp(-clip, clip)

    # ② 算每步的功劳权重
    # total = 两步质量之和 [K, 1, N]
    total = stage_quality.sum(1, keepdim=True)
    # 权重 = 2 × 本步质量 / 两步总质量
    # 为什么乘 2？因为两步权重加起来应该等于 2（这样 advantage 分摊后均值不变）
    # 例如两步质量都是 0.5，各得权重 1；第一步 0.8、第二步 0.2，各得 1.6 和 0.4
    # 兜底：如果两步质量都是 0，就平均分配（各得 1）
    weights = torch.where(total > 0, 2 * stage_quality / total.clamp_min(1e-12),
                          torch.ones_like(stage_quality))

    # ③ 把优势按权重分到两步
    allocated = advantage[:, None] * weights

    # ④ 守恒校验：两步分配后的均值必须等于原优势
    # （因为权重均值为 1，所以乘完取均值应该还原）
    if (allocated.mean(1) - advantage).abs().amax() > 1e-6:
        raise RuntimeError("terminal credit conservation failed")
    return advantage, weights, allocated
