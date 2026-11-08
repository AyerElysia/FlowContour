"""所有 FlowContour 入口共用的一份科学协议（scientific contract）。

训练、推理、恢复检查点、安装校验都会读取这里的常量，并与
`protocol_metadata()` 做精确字典比对。改任一字段都会让旧检查点
无法 resume / infer，因此不要在这里做“兼容性放宽”。
"""

# 当前主线身份。检查点、manifest、推理日志都带这个字符串。
PROTOCOL_VERSION = "flowcontour.raos3.ordered.v2"

# 两步外层演化的阶段条件 s。
# s 为累计完成进度：首次更新取 0，第二次更新取 0.6667。
STAGE_VALUES = (0.0, 0.6667)

# 每个外层阶段内部的 ODE 求值次数（NFE per stage）。
# 两步合计 total_nfe = 2 * ODE_STEPS = 8。
ODE_STEPS = 4

# 闭合轮廓离散点数。网络、目标构造、奖励、推理接口都按这个规格对齐。
CONTOUR_POINTS = 128

# 区域奖励用的 Normalized Surface Dice 容差（单位：网络输入像素）。
# 1/2/4/8 px 等权平均；坐标空间见 metadata 的 reward_coordinate_space。
NSD_SCALES_PX = (1.0, 2.0, 4.0, 8.0)


def protocol_metadata():
    """序列化后写入检查点 / manifest，供入口做身份校验。"""
    return {
        "version": PROTOCOL_VERSION,
        "contour_points": CONTOUR_POINTS,
        "stage_values": list(STAGE_VALUES),
        # s 与监督状态的实际 GT 插值进度相同。
        "stage_semantics": "cumulative_fraction_0_then_0.6667",
        # True：每步都执行完整预测位移 Δ，不乘外层 frac。
        "full_displacement": False,
        "outer_fractions": [0.6667, 1.0],
        # 部署与 GDPO 轨迹都是两步完整动作（不是旧版五阶段）。
        "outer_steps": 2,
        # 内层 ODE 求解器：二阶 Adams-Bashforth。
        "solver": "ab2",
        "nfe_per_stage": ODE_STEPS,
        "total_nfe": 2 * ODE_STEPS,
        # 潜变量尺度固定为 1.0，与训练采样一致。
        "latent_scale": 1.0,
        "latent_covariance": "cyclic_binomial5_unit_variance_plus_0.2_white",
        "displacement_normalization": "current_contour_max_axis_span_min1",
        "supervised_state_sampling": "v4_10_continuous_gaussian_mixture",
        "supervised_fm_objective": "curvature_velocity_mse_endpoint_mse_soft_chamfer",
        # 模型前向计算精度；与权重存储 dtype 无关。
        "model_compute_dtype": "float32",
        "reward_scales_input_px": list(NSD_SCALES_PX),
        # 奖励在网络输入像素坐标系上计算，不是原始 CT 毫米空间。
        "reward_coordinate_space": "network_input_pixels",
        # GDPO 信用：最终轨迹优势 × 该阶段平均单步质量，而不是各阶段独立归一化。
        "credit": "terminal_advantage_times_mean_one_stage_quality",
    }
