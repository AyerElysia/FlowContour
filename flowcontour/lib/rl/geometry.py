"""GDPO 强化学习里的"几何探索"工具。

一句话：网络预测了一个轮廓位移 mean 之后，GDPO 会在它基础上加一点
"随机扰动"，故意走出不一样的轮廓，用来试哪种形状更好看（NSD 分数更高）。
本文件负责：

    ① 算扰动该往哪个方向加（法向）
    ② 生成几种固定形状的扰动（低频波浪 or 局部鼓包）
    ③ 给扰动打分（logprob），后面 PPO 更新要用

为什么要这么做？Flow 本身已经是生成式（潜变量 z 随机），但 GDPO 还想
在动作层面再叠一层结构化的几何探索——专门探索"轮廓的法向起伏"这种
对医疗分割最有意义的方向，而不是乱挪。
"""

import math
import torch


# ═══════════════════════════════════════════════════════════════
# ① 算每个点的法向（轮廓垂直方向）
# ═══════════════════════════════════════════════════════════════
def contour_normals(poly):
    """给闭合轮廓上每个点，算一个"朝外的方向"（单位法向量）。

    为什么要法向？因为轮廓修改最有意义的方式是沿法向"鼓起或凹陷"，
    而不是沿切线方向滑动（滑动基本不改变形状，纯浪费动作）。

    输入 poly：[N, 128, 2]，N 条轮廓、每条 128 个点、每点 (x, y)。
    输出      ：[N, 128, 2]，每个点一个长度为 1 的方向向量。
    """
    # 切向 = 下一个点 − 上一个点。roll 把数组循环移位，所以最末点接最首点。
    tangent = torch.roll(poly, -1, 1) - torch.roll(poly, 1, 1)
    # 坑：如果某段两个顶点重合（退化边），切向就是 0，没法算法向。
    # 修复策略：把"上一个点"挪到再往前一个、再往前一个……直到找到不重合的。
    for offset in range(2, poly.shape[1]):
        bad = tangent.norm(dim=-1) < 1e-8          # 找出长度为 0 的坏切向
        if not bad.any():
            break                                    # 没有坏的就结束
        alternative = torch.roll(poly, -offset, 1) - poly
        tangent = torch.where(bad[..., None], alternative, tangent)
    # 法向 = 切向顺时针转 90°：切向 (a, b) → 法向 (−b, a)。
    # 这里选的"外法线"方向取决于轮廓绕向，只要整条轮廓统一就行。
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
    # 兜底：如果整条轮廓都塌缩成一个点，给一个固定方向 (1, 0)，防止除零。
    normal = torch.where((normal.norm(dim=-1) < 1e-8)[..., None],
                         normal.new_tensor([1.0, 0.0]), normal)
    # 除以自身长度，变成单位向量（长度为 1）。
    return normal / normal.norm(dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════════
# ② 准备一组"波浪形状"的基底（傅里叶基）
# ═══════════════════════════════════════════════════════════════
def fourier_basis(points, modes, device, dtype):
    """在 128 个点的闭合轮廓上，造出 modes 种"波浪形状"。

    想象把轮廓的 128 个点编号 0~127，绕一圈。
    这里造一组标准波形：
        第 0 列：全部相等（直流，整体平移）
        第 1 列：cos，1 个周期（一圈起伏 1 次）
        第 2 列：sin，1 个周期
        第 3 列：cos，2 个周期
        第 4 列：sin，2 个周期
        ...
    每种波形是一个 128 维向量，最后沿法向"叠加"就形成轮廓起伏。

    为什么用傅里叶？因为低频（少几个周期）的形变才像真实器官的边缘，
    高频（几十个周期）会变成锯齿，不像真实轮廓。这里只取前 modes 个低频。
    """
    if not 1 <= modes < points:
        raise ValueError("Fourier modes must lie in [1, point_count)")
    # 角度：0, 2π/128, 4π/128, ... 共 128 个
    theta = torch.arange(points, device=device, dtype=dtype) * (2 * math.pi / points)
    # 第一列：全 1/√128（直流分量）
    columns = [torch.ones_like(theta) / math.sqrt(points)]
    frequency = 1
    while len(columns) < modes:
        columns.append(math.sqrt(2 / points) * torch.cos(frequency * theta))
        if len(columns) < modes:
            columns.append(math.sqrt(2 / points) * torch.sin(frequency * theta))
        frequency += 1
    # 拼成 [128, modes] 的矩阵，每列是一种波形
    return torch.stack(columns, -1)


# ═══════════════════════════════════════════════════════════════
# ③ 构造一张"几何图"：决定这次扰动由哪几种波形叠加而成
# ═══════════════════════════════════════════════════════════════
def geometry_component(state, sigma, modes=8, local=False,
                       widths=(8, 12), width_probabilities=(.3, .7)):
    """为每条轮廓准备一套"扰动模板"，返回三样东西：

        design  ：[N, 256, m] 矩阵，m 种波形。用它乘随机系数就能得到扰动。
        inverse ：design 的伪逆，用来反推"这个动作是用了哪些系数"。
        offset  ：[N, 128, 2]，可选的固定偏移（local 模式时是一个鼓包）。

    参数：
        state ：当前轮廓 [N, 128, 2]。
        sigma ：扰动的"幅度"（标准差），已经从输入像素换算成特征像素。
        modes ：用几种低频波形（默认 8，即 0~3 次 cos/sin）。
        local ：True 时额外加一个随机"鼓包/凹坑"，模拟局部边界不确定。

    通俗类比：design 就像一盒蜡笔，每支蜡笔是一种波浪形状；
    后面 sample 时随机抽一组颜色系数，把这些蜡笔在法向上画出来，
    就得到一个完整的扰动形状。
    """
    n, p, _ = state.shape
    # ① 先造 8 种低频波形，整体幅度乘以 sigma
    base = fourier_basis(p, modes, state.device, torch.float64)
    chart = (base * float(sigma)).expand(n, -1, -1).clone()
    offset = state.new_zeros((n, p), dtype=torch.float64)
    info = {"local": bool(local), "sigma_feature": float(sigma)}

    # ② 如果开了 local，再加一个"单峰/单谷"的鼓包
    if local:
        # 鼓包中心：在 128 个点里随机挑一个
        center = torch.randint(p, (n,), device=state.device)
        # 鼓包宽度：随机选 8 或 12（按 30%/70% 概率）
        choice = torch.multinomial(torch.tensor(width_probabilities, device=state.device), n, True)
        width = torch.tensor(widths, device=state.device)[choice].double()
        # 算每个点到中心的"沿轮廓弧长距离"（按比例，0~0.5）
        edges = (torch.roll(state.double(), -1, 1) - state.double()).norm(dim=-1)
        perimeter = edges.sum(1, keepdim=True)
        if (perimeter <= 1e-12).any():
            raise ValueError("a local bump needs a noncollapsed contour")
        # 每个点沿轮廓的累积位置占比 [0, 1)
        position = (edges.cumsum(1) - edges) / perimeter
        center_position = position.gather(1, center[:, None])
        fraction = (position - center_position).abs()
        # 用 min(frac, 1-frac) 处理"绕一圈"的最短距离
        distance = torch.minimum(fraction, 1 - fraction) * 128
        # 余弦窗：中心最高、两边平滑降到 0（这样鼓包边缘不突兀）
        bump = .5 * (1 + torch.cos(math.pi * (distance / (width[:, None] / 2)).clamp(max=1)))
        # 高度方向 50% 概率朝外鼓、50% 朝内凹
        sign = torch.where(torch.rand(n, device=state.device) < .5, -1., 1.)
        # 高度值：70% 选 2px，25% 选 3px，5% 选 4px（小概率大扰动增加探索）
        height_index = torch.multinomial(state.new_tensor([.7, .25, .05]), n, True)
        height_px = state.new_tensor([2., 3., 4.])[height_index] * sign
        # 输入像素 → 特征像素（除以 4，因为 down_ratio=4）
        offset = bump * (height_px.double() / 4)[:, None]
        # 鼓包本身也作为一种"形状"加入 chart，方便后面算 logprob
        chart = torch.cat((chart, (bump / 4)[..., None]), -1)
        info.update(center=center.tolist(), chosen_width_reference=width.tolist(),
                    height_center_px=height_px.tolist())

    # ③ 把"形状 × 法向"组合成最终的扰动基底
    # chart 是 [N, 128, m]，normals 是 [N, 128, 2]
    # design[n, 2i, k] = chart[n, i, k] * normal_x，design[n, 2i+1, k] = chart * normal_y
    # 也就是每个形状沿法向投影到 (x, y) 两个坐标上
    normals = contour_normals(state.double())
    design = (chart[..., None] * normals[:, :, None, :]).permute(0, 1, 3, 2).flatten(1, 2)
    # ④ 求伪逆：后面给一个实际动作，就能反推出"它对应了哪些系数"
    inverse = torch.linalg.pinv(design, rtol=1e-12)
    # 校验：inverse @ design 应该接近单位矩阵，否则这组基底退化了
    identity_error = (inverse @ design - torch.eye(chart.shape[-1], device=state.device)).abs().amax()
    if not torch.isfinite(inverse).all() or identity_error > 1e-6:
        raise RuntimeError(f"degenerate geometry chart: {float(identity_error)}")
    # detach：这些是"冻结的几何结构"，不参与梯度回传，只当常数用
    return {"design": design.detach(), "inverse": inverse.detach(),
            "offset": (offset[..., None] * normals).detach(), "info": info}


# ═══════════════════════════════════════════════════════════════
# ④ 给一个动作打分：它在这套几何图里出现的"概率"有多大
# ═══════════════════════════════════════════════════════════════
def chart_logprob(action, mean, component):
    """把动作投影到几何图的坐标上，然后当标准正态算 log 概率。

    通俗说：网络输出了 mean，我们加了扰动得到 action。
    PPO 时需要知道"这个 action 是按什么概率采出来的"，才能算 ratio。

    数学：
        residual = action − mean − offset          # 纯扰动部分
        c = inverse @ residual                      # 反推出对应的 m 个系数
        logprob = Σ (−0.5 * cᵢ² − 0.5·ln(2π))       # 标准正态的 log 概率
    """
    residual = (action.double() - mean.double() - component["offset"]).flatten(1)
    coefficients = torch.einsum("nmp,np->nm", component["inverse"], residual)
    return (-.5 * coefficients.square() - .5 * math.log(2 * math.pi)).mean(-1)


def chart_kl(mean, reference_mean, component):
    """当前策略的 mean 和"冻结参考策略"的 reference_mean 之间的 KL 散度。

    为什么需要？GDPO 更新时怕策略一下跑太远，所以用 KL 当正则项惩罚
    偏移过大的更新。KL 越大说明偏离参考策略越远。

    注意只比较 mean 的差（协方差假设相同），所以就是 0.5 * ||位移||²。
    """
    shift = (mean.double() - reference_mean.detach().double()).flatten(1)
    coordinates = torch.einsum("nmp,np->nm", component["inverse"], shift)
    return .5 * coordinates.square().mean(-1)


# ═══════════════════════════════════════════════════════════════
# ⑤ 主入口：真正采一个带几何扰动的动作
# ═══════════════════════════════════════════════════════════════
def sample_geometry(mean, state, sigma, *, local=False, modes=8):
    """在网络预测的 mean 基础上，加一个结构化几何扰动，返回动作。

    流程：
        1. 构造几何图（哪些波形可用）
        2. 给每个波形抽一个标准正态系数 z
        3. 扰动 = design @ z
        4. 动作 = mean + 扰动 + offset
        5. 立刻用 chart_logprob 反算一遍概率，确认和直接从 z 算的一致
           （防止数值误差导致 PPO 的 ratio 算错）
    """
    component = geometry_component(state, sigma, modes=modes, local=local)
    # 抽 m 个标准正态系数，每个系数控制一种波形的强弱
    z = torch.randn(state.shape[0], component["design"].shape[-1],
                    device=state.device, dtype=torch.float64)
    # design: [N, 256, m]，z: [N, m] → 扰动 [N, 256] → reshape 成 [N, 128, 2]
    perturbation = torch.einsum("npm,nm->np", component["design"], z).view_as(state)
    # 最终动作 = 网络均值 + 随机扰动 + 固定偏移（local 鼓包）
    action = (mean.double() + perturbation + component["offset"]).to(state.dtype)
    # 自检：直接从 z 算的 logprob 应该等于用 chart_logprob 反算的结果
    # 这是"概率可回放"的校验，保证 PPO 里 old/new logprob 对得上
    expected = (-.5 * z.square() - .5 * math.log(2 * math.pi)).mean(-1)
    replay = chart_logprob(action, mean, component)
    error = (replay - expected).abs().amax()
    if not torch.isfinite(error) or error >= .002:
        raise RuntimeError(f"geometry probability replay failed: {float(error)}")
    component["replay_error"] = float(error)
    # 返回：动作（detach，不回传梯度）、logprob、几何图结构（供 replay 用）
    return action.detach(), replay.detach(), component
