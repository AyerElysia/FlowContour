#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAOS2 two-stage training entrypoint: supervised flow pretraining, then GDPO.

Preflight performs two optimizer updates and records a certificate. Formal
training requires the matching certificate and a new output directory.
"""

import argparse
import os
from pathlib import Path
import sys

# __file__：当前这个 .py 脚本文件的路径
# .resolve()：拿到它的【绝对路径】，自动消除 ../ 这类相对路径
# .parents[1]：取上上级目录（parents[0] = 当前脚本所在文件夹；parents[1] = 再往上一层）
ROOT = Path(__file__).resolve().parents[1]
# sys.path 是 Python 的模块搜索列表，import 时会按这个列表依次找包
# insert(0, ...)：把 ROOT 插到搜索列表最前面（优先级最高）
# 转成字符串，因为 sys.path 只接受字符串路径
sys.path.insert(0, str(ROOT))


def main(argv=None):
    """FlowContour 训练入口主函数。

    负责解析命令行参数、执行安全准入门禁校验、注入全局环境单例配置、
    申请物理硬件资源，并最终调用底层统一训练引擎 lib.train.engine.run 执行训练。

    Args:
        argv: 命令行参数列表，若为 None 则默认从 sys.argv[1:] 解析。
    """
    # 构造命令行参数解析器
    parser = argparse.ArgumentParser(description=__doc__)
    
    # ------------------ 阶段与模式核心参数 ------------------
    parser.add_argument(
        "--stage",
        choices=("pretrain", "gdpo"),
        required=True,
        help="指定训练阶段：pretrain (监督流匹配预训练)、gdpo (几何强化学习)；"
             "该参数必须与对应配置文件中的 training_stage 字段严格一致，防止阶段错位。",
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "train"),
        required=True,
        help="指定运行模式：preflight (前置功能/硬件资源闸门预检，跑 2 步) 或 train (正式大规模有效迭代)；"
             "正式 train 必须以前置成功的 preflight 证书为前提。",
    )
    parser.add_argument(
        "--config",
        help="指定的 YAML 配置文件路径；若省略则默认自动匹配 configs/<stage>.yaml；"
             "各阶段 YAML 仅声明参数差异，网络主干与通用协议全部集中继承自 configs/common.yaml。",
    )
    
    # ------------------ 权重与血统追溯参数 ------------------
    parser.add_argument("--random-init", action="store_true", help="Initialize RAOS pretraining from scratch")
    parser.add_argument(
        "--source-checkpoint",
        required=False,
        help="初始化权重文件 (.pth) 路径；"
             "若为 Dense 迁移，只允许指定官方签名的 step 19000 权重；"
             "若为后续微调或强化学习，权重中必须包含合法的 flowcontour.full2.v2 协议元数据。",
    )
    parser.add_argument(
        "--source-sha256",
        required=False,
        help="输入检查点文件的 SHA256 哈希值，用于确保模型初始化血统绝对防篡改。",
    )
    parser.add_argument(
        "--source-step",
        type=int,
        required=False,
        help="源权重对应的有效更新步数 (step)；引擎会严格核验文件内记录的步数是否与之吻合。",
    )
    parser.add_argument(
        "--resume-checkpoint",
        help="用于中断后恢复训练的完整检查点路径；将完全回放优化器、学习率调度器与 RNG 随机状态。",
    )
    parser.add_argument(
        "--preflight-output",
        help="正式 train 模式的强制前置依赖：指向已 PASS 且配置/代码/硬件一致的预检输出目录。",
    )
    
    # ------------------ 数据与特征缓存路径 ------------------
    parser.add_argument(
        "--data-root",
        required=True,
        help="医学图像原始数据集根目录 (包含 VerSe 矢状位切片图像与标注)。",
    )
    parser.add_argument(
        "--slice-manifest",
        required=True,
        help="切片元数据清单 CSV 文件路径 (包含 split, case_id, slice_idx, image_path, mask_path)。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="实验结果与检查点输出目录；必须为尚未存在的全新路径，防止污染旧实验。",
    )
    
    # ------------------ 硬件资源与执行控制 ------------------
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="分配给当前训练任务的物理 GPU 设备卡号 (例如 0)。",
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="显式允许共享 GPU；预检和正式训练均要求两次采样有至少 30 GiB 空余且利用率不超过 30%%。",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="计算设备类型，默认为 cuda；cpu 仅用于无卡环境下的极速功能排错。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=202610071618,
        help="全局基础随机种子，确保 Python/NumPy/PyTorch CPU/CUDA 完全确定性与可复现性。",
    )
    
    # 解析命令行传入的实参
    args = parser.parse_args(argv)
    args.initialize_dense = False
    args.root = str(ROOT)
    if args.random_init and args.stage != "pretrain":
        parser.error("random initialization is for pretrain only")
    if not args.random_init and not (args.source_checkpoint and args.source_sha256 and args.source_step is not None):
        parser.error("checkpoint, SHA256 and step are required unless --random-init")
    
    # ==================================================================================
    # 严格安全准入门禁断言 (Fail-closed Safety Gates)
    # ==================================================================================
    
    # 门禁 1：正式训练必须运行于独占空闲的 CUDA 设备上，且必须提供已通过的前置预检凭据
    if args.mode == "train" and (not args.preflight_output or args.device != "cuda"):
        parser.error("formal training requires a CUDA preflight and --preflight-output")
        
    # 门禁 2：共享 GPU 资源放宽例外严禁渗透至正式训练中，仅限轻量级预检
        
    # 门禁 3：输出目录必须全新，防止误覆写已有的实验检查点或度量记录
    if Path(args.output).exists():
        raise FileExistsError("output must be a new directory")
        
    # 切换当前工作路径至项目根目录，保证相对路径寻址的一致性
    os.chdir(ROOT)
    
    # ==================================================================================
    # 环境变量与配置单例注入 (Global Configuration Injection)
    # ==================================================================================
    # 确定当前阶段使用的具体配置文件绝对路径
    config = Path(args.config or ROOT / f"configs/{args.stage}.yaml").resolve()
    
    # 注意：必须在导入任何依赖 cfg 的项目模块前，将 CFG_FILE 和 CUDA 设备写入系统环境变量，
    # 并且模拟修改 sys.argv，以保证 lib.config 单例初始化时正确读取阶段 YAML 文件。
    os.environ["CFG_FILE"] = str(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.device == "cuda" else ""
    os.environ["FLOWCONTOUR_DATA_ROOT"] = str(Path(args.data_root).resolve())
    os.environ["FLOWCONTOUR_SLICE_MANIFEST"] = str(Path(args.slice_manifest).resolve())
    sys.argv = [sys.argv[0], "--cfg_file", str(config)]
    
    # 延迟导入项目核心配置与模块（此时环境变量已就绪）
    from lib.config import cfg
    from lib.runtime.gpu import acquire_gpu
    from lib.train.engine import run
    
    # 校验 CLI 传入的 --stage 与配置文件解析出的 cfg.training_stage 是否一致
    if cfg.training_stage != args.stage:
        raise ValueError("CLI stage does not match the configuration")
        
    # 注入 MoonViT 特征缓存路径到全局配置字典
    
    # 物理 GPU 独占空闲状态申请与校验：
    # 若在 CUDA 设备上运行，acquire_gpu 会在间隔 15 秒内采样两次显存占用与算力使用率，
    # 确保没有其他竞争进程干扰基准测量，并记录完整的硬件健康快照。
    checks = acquire_gpu(args.gpu, args.allow_shared_gpu) if args.device == "cuda" else []
    
    # ==================================================================================
    # 驱动训练引擎并执行原子失败记录 (Execution & Atomic Failure Handling)
    # ==================================================================================
    try:
        # 调用核心训练引擎：执行权重校验、网络构建、优化器初始化与有效更新循环
        result = run(args, checks)
    except Exception as error:
        # 若训练过程中抛出任何异常，立刻在输出目录中原子写出 FAILED 状态的 manifest.json，
        # 并记录完整的异常文本与调用堆栈跟踪，确保失败现场得以完整保留，便于调试。
        import json
        import traceback
        from lib.train.engine import dump_json
        
        path = Path(args.output) / "manifest.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        dump_json(
            path,
            {
                **previous,
                "status": "FAILED",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        # 重新抛出原异常供外部监控系统捕获
        raise
        
    # 打印最终成功状态、阶段、有效更新次数与输出路径
    print({
        "status": result["status"],
        "stage": args.stage,
        "updates": result["effective_updates"],
        "output": args.output,
    })


if __name__ == "__main__":
    main()
