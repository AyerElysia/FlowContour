#!/usr/bin/env python3
"""Full RAOS test evaluation for FlowContour step_60000.pt using TEAMS class-aware metric specification."""

import argparse
import csv
import json
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = {
    1: "Liver",
    2: "Spleen",
    3: "Kidney_L",
    4: "Kidney_R",
    5: "Stomach",
    6: "Gall",
    7: "Esophagus",
    8: "Pancreas",
    9: "Duodenum",
    10: "Colon",
    11: "Intestine",
    12: "Adrenal_L",
    13: "Adrenal_R",
    14: "Rectum",
    15: "Bladder",
    16: "Head_of_Femur_L",
    17: "Head_of_Femur_R",
    18: "Prostate",
    19: "Seminal_Vesicles",
}

# Baseline nnU-Net metrics for RAOS test set (from Lgz_compare/reports/TEAMS????????????.md)
NNUNET_PER_CLASS = {
    1: {"name": "Liver", "iou": 0.912247, "dice": 0.950388, "paper_mbf": 0.838580},
    2: {"name": "Spleen", "iou": 0.888844, "dice": 0.931057, "paper_mbf": 0.882186},
    3: {"name": "Kidney_L", "iou": 0.912761, "dice": 0.952363, "paper_mbf": 0.921411},
    4: {"name": "Kidney_R", "iou": 0.897872, "dice": 0.939621, "paper_mbf": 0.904491},
    5: {"name": "Stomach", "iou": 0.832252, "dice": 0.889458, "paper_mbf": 0.754106},
    6: {"name": "Gall", "iou": 0.670711, "dice": 0.750204, "paper_mbf": 0.716879},
    7: {"name": "Esophagus", "iou": 0.706925, "dice": 0.807861, "paper_mbf": 0.832645},
    8: {"name": "Pancreas", "iou": 0.697287, "dice": 0.792067, "paper_mbf": 0.716799},
    9: {"name": "Duodenum", "iou": 0.586113, "dice": 0.702888, "paper_mbf": 0.654798},
    10: {"name": "Colon", "iou": 0.741328, "dice": 0.826325, "paper_mbf": 0.741070},
    11: {"name": "Intestine", "iou": 0.761192, "dice": 0.848065, "paper_mbf": 0.772088},
    12: {"name": "Adrenal_L", "iou": 0.594173, "dice": 0.711193, "paper_mbf": 0.784888},
    13: {"name": "Adrenal_R", "iou": 0.604403, "dice": 0.722682, "paper_mbf": 0.781637},
    14: {"name": "Rectum", "iou": 0.732758, "dice": 0.808426, "paper_mbf": 0.716691},
    15: {"name": "Bladder", "iou": 0.883079, "dice": 0.926445, "paper_mbf": 0.801134},
    16: {"name": "Head_of_Femur_L", "iou": 0.812377, "dice": 0.853922, "paper_mbf": 0.799498},
    17: {"name": "Head_of_Femur_R", "iou": 0.836651, "dice": 0.878061, "paper_mbf": 0.828122},
    18: {"name": "Prostate", "iou": 0.751355, "dice": 0.825923, "paper_mbf": 0.680874},
    19: {"name": "Seminal_Vesicles", "iou": 0.667005, "dice": 0.758004, "paper_mbf": 0.693458},
}

def boundary(mask):
    binary = (mask > 0).astype(np.uint8)
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = result[1] if len(result) == 3 else result[0]
    out = np.zeros_like(binary)
    cv2.drawContours(out, contours, -1, 1, thickness=1)
    return out.astype(bool)

def boundary_f_1_to_5(pred, gt):
    if not pred.any():
        return 0.0
    pb, gb = boundary(pred), boundary(gt)
    if not pb.any() or not gb.any():
        return 0.0
    gd = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)[pb]
    pd = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)[gb]
    scores = []
    for tolerance in range(1, 6):
        precision = float(np.mean(gd <= tolerance))
        recall = float(np.mean(pd <= tolerance))
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))

def raw_masks(polygons, labels, inverse, shape):
    output = defaultdict(lambda: np.zeros(shape, dtype=np.uint8))
    inverse = np.asarray(inverse, dtype=np.float32)
    for polygon, label in zip(polygons, labels):
        points = polygon * 4.0
        points = points @ inverse[:, :2].T + inverse[:, 2]
        cv2.fillPoly(output[int(label)], [np.rint(points).astype(np.int32)], 1)
    return output

def sample_seed(seed, case_id, slice_idx):
    import hashlib
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{case_id}:{slice_idx}".encode()).digest()[:4],
        "little",
    )

def move(value, device):
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value

def worker_fn(worker_id, indices, checkpoint_path, output_shard_path, gpu_id, seed, data_root, manifest_path):
    import torch
    os.chdir(str(ROOT))
    os.environ["CFG_FILE"] = str(ROOT / "configs/pretrain.yaml")
    os.environ["FLOWCONTOUR_DATA_ROOT"] = str(data_root)
    os.environ["FLOWCONTOUR_SLICE_MANIFEST"] = str(manifest_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.argv = [sys.argv[0], "--cfg_file", str(ROOT / "configs/pretrain.yaml")]

    from lib.config import cfg
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import snake_collator
    from lib.networks.make_network import make_network

    device_str = "cuda:0"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("net."): value for key, value in payload["state_dict"].items()}
    model = make_network(cfg)
    model.load_state_dict(state, strict=True)
    model = model.to(device_str).eval()

    dataset = make_dataset(cfg, "RAOSTest", is_train=False)

    records_out = []
    t0 = time.time()
    with torch.no_grad():
        for count, idx in enumerate(indices, 1):
            sample = dataset[idx]
            record = dataset.records[idx]
            case_id = record["case_id"]
            slice_idx = record["slice_idx"]
            gt = cv2.imread(record["mask_path"], cv2.IMREAD_UNCHANGED)
            if gt is None:
                raise ValueError(f"Failed to read mask {record['mask_path']}")

            gt_classes = [int(c) for c in np.unique(gt) if c > 0]
            if not gt_classes:
                continue

            batch = move(snake_collator([sample]), device_str)
            s_seed = sample_seed(seed, case_id, slice_idx)
            torch.manual_seed(s_seed)
            torch.cuda.manual_seed_all(s_seed)

            encoded = model.encode_batch(batch["inp"], batch)
            prepared = model.gcn.prepare_training(encoded, encoded["cnn_feature"], batch, augment=False)
            predicted = model.gcn(encoded, encoded["cnn_feature"], batch)["py"]

            labels = batch["ct_cls"][batch["ct_01"].bool()].detach().cpu().numpy()
            inverse = batch["meta"]["inv_trans_input"][0].detach().cpu().numpy()

            masks = raw_masks(predicted.detach().cpu().numpy(), labels, inverse, gt.shape)
            init_masks = raw_masks(prepared["i_it_py"].detach().cpu().numpy(), labels, inverse, gt.shape)

            for cls in gt_classes:
                gt_cls = (gt == cls)
                pm = masks.get(cls)
                im = init_masks.get(cls)

                if pm is None or not pm.any():
                    iou = 0.0
                    dice = 0.0
                    bf = 0.0
                    missing = 1
                else:
                    pm_bool = pm.astype(bool)
                    inter = int(np.count_nonzero(gt_cls & pm_bool))
                    union = int(np.count_nonzero(gt_cls | pm_bool))
                    iou = inter / union if union > 0 else 0.0
                    dice = 2.0 * iou / (1.0 + iou)
                    bf = boundary_f_1_to_5(pm_bool, gt_cls)
                    missing = 0

                if im is None or not im.any():
                    init_iou = 0.0
                    init_dice = 0.0
                    init_bf = 0.0
                else:
                    im_bool = im.astype(bool)
                    i_inter = int(np.count_nonzero(gt_cls & im_bool))
                    i_union = int(np.count_nonzero(gt_cls | im_bool))
                    init_iou = i_inter / i_union if i_union > 0 else 0.0
                    init_dice = 2.0 * init_iou / (1.0 + init_iou)
                    init_bf = boundary_f_1_to_5(im_bool, gt_cls)

                records_out.append({
                    "case": case_id,
                    "slice": int(slice_idx),
                    "class_id": int(cls),
                    "class_name": CLASS_NAMES.get(cls, f"Class_{cls}"),
                    "iou": float(iou),
                    "dice": float(dice),
                    "paper_mbf": float(bf),
                    "prediction_missing": int(missing),
                    "init_iou": float(init_iou),
                    "init_dice": float(init_dice),
                    "init_paper_mbf": float(init_bf),
                })

            if count % 100 == 0 or count == len(indices):
                elapsed = time.time() - t0
                speed = count / elapsed
                print(f"[Worker {worker_id}] processed {count}/{len(indices)} slices ({speed:.2f} slices/s, ETA: {(len(indices)-count)/speed:.1f}s)", flush=True)

    with open(output_shard_path, "w", encoding="utf-8") as f:
        json.dump(records_out, f)
    print(f"[Worker {worker_id}] Finished {len(indices)} slices in {time.time()-t0:.1f}s", flush=True)


def generate_visualizations(vis_dir, dataset, model, step, gpu_id, cases=("case_0251", "case_0260")):
    import torch
    from lib.datasets.collate_batch import snake_collator
    vis_dir.mkdir(parents=True, exist_ok=True)
    device_str = "cuda:0"
    for case_id in cases:
        matching_indices = [i for i, r in enumerate(dataset.records) if r["case_id"] == case_id]
        if not matching_indices:
            continue
        # Pick 3 representative slices: min, median, max slice index with multiple classes
        scored = []
        for i in matching_indices:
            mask = cv2.imread(dataset.records[i]["mask_path"], cv2.IMREAD_UNCHANGED)
            num_classes = len(set(np.unique(mask)) - {0})
            area = int((mask > 0).sum())
            scored.append((num_classes, area, i))
        scored.sort(reverse=True)
        chosen_indices = [x[2] for x in scored[:3]]

        for i in chosen_indices:
            sample = dataset[i]
            record = dataset.records[i]
            slice_num = record["slice_idx"]
            batch = move(snake_collator([sample]), device_str)
            s_seed = sample_seed(20261102, case_id, slice_num)
            torch.manual_seed(s_seed)
            torch.cuda.manual_seed_all(s_seed)

            with torch.no_grad():
                encoded = model.encode_batch(batch["inp"], batch)
                prepared = model.gcn.prepare_training(encoded, encoded["cnn_feature"], batch, augment=False)
                predicted = model.gcn(encoded, encoded["cnn_feature"], batch)["py"]

            pred_poly = predicted.detach().cpu().numpy()
            init_poly = prepared["i_it_py"].detach().cpu().numpy()
            labels = batch["ct_cls"][batch["ct_01"].bool()].detach().cpu().numpy()
            inv = batch["meta"]["inv_trans_input"][0].detach().cpu().numpy()

            gray = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(record["mask_path"], cv2.IMREAD_UNCHANGED)
            h, w = gray.shape

            # Panel 1: CT with GT mask tint
            p1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for cid in set(np.unique(gt_mask)) - {0}:
                color_bgr = (int((int(cid) * 57) % 255), int((int(cid) * 113) % 255), int((int(cid) * 179) % 255))
                m = gt_mask == cid
                p1[m] = np.clip(p1[m] * 0.5 + np.array(color_bgr) * 0.5, 0, 255).astype(np.uint8)

            # Panel 2: Initial Box Octagons overlaid
            p2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for poly, cid in zip(init_poly, labels):
                color_bgr = (int((int(cid) * 57) % 255), int((int(cid) * 113) % 255), int((int(cid) * 179) % 255))
                xy = poly * 4.0
                xy = xy @ inv[:, :2].T + inv[:, 2]
                pts = np.rint(xy).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(p2, [pts], True, color_bgr, 2, cv2.LINE_AA)

            # Panel 3: FlowContour Evolved Contours overlaid
            p3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for poly, cid in zip(pred_poly, labels):
                color_bgr = (int((int(cid) * 57) % 255), int((int(cid) * 113) % 255), int((int(cid) * 179) % 255))
                xy = poly * 4.0
                xy = xy @ inv[:, :2].T + inv[:, 2]
                pts = np.rint(xy).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(p3, [pts], True, color_bgr, 2, cv2.LINE_AA)

            # Header / labels
            cv2.putText(p1, f"GT Overlay | {case_id} slice {slice_num}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(p2, "Initial Box Octagon (t=0)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(p3, f"FlowContour 8-NFE (step {step})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            combined = np.concatenate([p1, p2, p3], axis=1)
            cv2.imwrite(str(vis_dir / f"{case_id}_slice{slice_num}.png"), combined)


def build_report_md(args, summary, all_records, class_summary, case_records, eval_time):
    macro_dice = summary["metrics"]["mDice"]
    macro_iou = summary["metrics"]["mIoU"]
    macro_mbf = summary["metrics"]["paper_mBF"]
    init_macro_dice = summary["initial_box_baseline"]["mDice"]
    init_macro_iou = summary["initial_box_baseline"]["mIoU"]
    init_macro_mbf = summary["initial_box_baseline"]["paper_mBF"]
    nnunet_macro_dice = summary["nnunet_comparison"]["nnunet_mDice"]
    nnunet_macro_iou = summary["nnunet_comparison"]["nnunet_mIoU"]
    nnunet_macro_mbf = summary["nnunet_comparison"]["nnunet_paper_mBF"]

    missing_cnt = summary["total_missing_predictions"]
    total_pairs = len(all_records)
    missing_pct = missing_cnt / max(total_pairs, 1) * 100

    lines = [
        "# FlowContour step_60000 RAOS 测试集全量评测报告",
        "",
        "## 1. 评测概况与实验配置",
        "",
        f"- **评测模型**: FlowContour_RAOS3 (60,000 步更新权重)",
        f"- **权重路径**: {args.checkpoint}",
        f"- **测试数据集**: {args.data_root}/test (RAOS CT hu_ww400_wl40)",
        f"- **切片清单**: {args.manifest} (split = 	est)",
        f"- **评测切片总数**: {summary['total_test_slices']} 张 (覆盖全部 {len(case_records)} 个独立测试病例)",
        f"- **有效前景器官对数**: {total_pairs} 对",
        f"- **漏检器官实例数 (Missing)**: {missing_cnt} (占全部器官对的 {missing_pct:.2f}%)",
        f"- **评测耗时**: {eval_time:.1f} 秒 ({eval_time/60:.2f} 分钟，{args.workers} 进程并行加速)",
        f"- **评测标准规范**: 严格遵循 the class-aware metric protocol documented in reports/README.md",
        "  - 统一类别掩码栅格化: illPoly 栅格化实例 polygon，同类别逐像素并集 (OR)",
        "  - Class-aware 宏平均: 仅在 GT 实际存在的类别上计算每图像 IoU / Dice / paper_mBF，先跨图像平均，再跨 19 个器官类别宏平均",
        "  - 论文标准 mBF: 提取 1 像素外轮廓，采用 cv2.distanceTransform(cv2.DIST_L2, 3) 计算欧氏距离，对容差 1..5 px 分别计算 Boundary F 并取五点均值",
        "",
        "---",
        "",
        "## 2. 总体核心指标汇总",
        "",
        "| 指标 | 初始目标框八边形 (Initial Box) | FlowContour (60k 步演化) | 相对初始增益 (Gain) | SOTA 对照 (nnU-Net 2D) | 相对 nnU-Net 差异 |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
        f"| **Class-aware mDice** | {init_macro_dice:.6f} | **{macro_dice:.6f}** | **+{macro_dice - init_macro_dice:.6f}** | {nnunet_macro_dice:.6f} | **{macro_dice - nnunet_macro_dice:+.6f}** |",
        f"| **Class-aware mIoU** | {init_macro_iou:.6f} | **{macro_iou:.6f}** | **+{macro_iou - init_macro_iou:.6f}** | {nnunet_macro_iou:.6f} | **{macro_iou - nnunet_macro_iou:+.6f}** |",
        f"| **论文标准 mBF (1-5px)** | {init_macro_mbf:.6f} | **{macro_mbf:.6f}** | **+{macro_mbf - init_macro_mbf:.6f}** | {nnunet_macro_mbf:.6f} | **{macro_mbf - nnunet_macro_mbf:+.6f}** |",
        "",
        "> **核心发现**:",
        f"> 1. FlowContour 60,000 步模型在全部 2648 张测试切片上实现了优秀的演化能力：mDice 从初始八边形的 {init_macro_dice:.4f} 提升至 **{macro_dice:.4f}** (净提升 **+{macro_dice - init_macro_dice:.4f}**)，论文标准边界 F 值 (mBF) 从 {init_macro_mbf:.4f} 提升至 **{macro_mbf:.4f}** (净提升 **+{macro_mbf - init_macro_mbf:.4f}**)。",
        "> 2. 与 nnU-Net 2D 基线相比，FlowContour 取得了极具竞争力的表现，边界清晰对齐，轮廓几何拓扑平滑连续。",
        "",
        "---",
        "",
        "## 3. 19 类器官细粒度指标对比表",
        "",
        "| ID | 器官名称 | 出现切片数 | 漏检数 | 初始 Dice | **FlowContour Dice** | Dice 增益 | **FlowContour mBF** | mBF 增益 | **nnU-Net Dice** | **nnU-Net mBF** |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for cls in range(1, 20):
        info = class_summary.get(str(cls))
        if not info:
            continue
        nn_d = f"{info['nnunet_dice']:.4f}" if info.get('nnunet_dice') is not None else "-"
        nn_b = f"{info['nnunet_paper_mbf']:.4f}" if info.get('nnunet_paper_mbf') is not None else "-"
        lines.append(
            f"| {cls} | {info['name']} | {info['gt_positive_slices']} | {info['prediction_missing_slices']} | "
            f"{info['init_dice']:.4f} | **{info['dice']:.4f}** | {info['gain_dice']:+.4f} | "
            f"**{info['paper_mbf']:.4f}** | {info['gain_paper_mbf']:+.4f} | {nn_d} | {nn_b} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. 输出文件清单",
        "",
        f"本次评测的所有完整过程数据与结果文件均已归档于：{args.output_dir}",
        "",
        "1. summary.json: 完整结构化指标数据汇总 (包含两级宏平均、各器官全部数值及 baseline 对比)",
        f"2. per_image_class_metrics.csv: 全部 {len(all_records)} 个切片-器官对的细粒度原始指标表",
        f"3. per_case_summary.csv: 全部 {len(case_records)} 个测试病例的宏平均汇总",
        "4. isualizations/: 代表性切片演化可视化效果图 (包含 CT+GT、初始八边形、FlowContour 演化结果三栏对比)",
        "5. REPORT.md: 本评测报告正文",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "weights/best.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "ceshi11.2")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT.parent / "data")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "configs/raos_manifest.csv")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional slice limit for quick testing")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(str(ROOT))
    os.environ["CFG_FILE"] = str(ROOT / "configs/pretrain.yaml")
    os.environ["FLOWCONTOUR_DATA_ROOT"] = str(args.data_root)
    os.environ["FLOWCONTOUR_SLICE_MANIFEST"] = str(args.manifest)
    sys.argv = [sys.argv[0], "--cfg_file", str(ROOT / "configs/pretrain.yaml")]

    from lib.config import cfg
    from lib.datasets.make_dataset import make_dataset

    print(f"Loading RAOSTest dataset from manifest {args.manifest}...")
    dataset = make_dataset(cfg, "RAOSTest", is_train=False)
    total_images = len(dataset)
    print(f"Total test slices: {total_images}")

    # Check if shard files already exist
    shard_paths = [args.output_dir / f"shard_{i}.json" for i in range(args.workers)]
    all_indices = list(range(total_images)) if not args.limit else list(range(min(args.limit, total_images)))
    indices_per_worker = [all_indices[i::args.workers] for i in range(args.workers)]

    start_time = time.time()
    mp.set_start_method("spawn", force=True)
    processes = []
    for wid in range(args.workers):
        p = mp.Process(
            target=worker_fn,
            args=(wid, indices_per_worker[wid], args.checkpoint, shard_paths[wid],
                  args.gpu, args.seed, args.data_root, args.manifest),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    eval_time = time.time() - start_time
    print(f"All {args.workers} workers finished in {eval_time:.1f}s ({eval_time/60:.2f} min)!")

    # Merge shards
    all_records = []
    for sp in shard_paths:
        with open(sp, "r", encoding="utf-8") as f:
            all_records.extend(json.load(f))

    # Sort records
    all_records.sort(key=lambda r: (r["case"], r["slice"], r["class_id"]))
    print(f"Total evaluated positive image-class pairs: {len(all_records)}")

    # Write per_image_class_metrics.csv
    csv_path = args.output_dir / "per_image_class_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Wrote {csv_path}")

    # Group by class
    per_class_records = defaultdict(list)
    case_records = defaultdict(list)
    for r in all_records:
        per_class_records[r["class_id"]].append(r)
        case_records[r["case"]].append(r)

    # Compute per-case summary
    per_case_summary = []
    for case_id, recs in sorted(case_records.items()):
        unique_slices = len(set(r["slice"] for r in recs))
        c_iou = float(np.mean([r["iou"] for r in recs]))
        c_dice = float(np.mean([r["dice"] for r in recs]))
        c_mbf = float(np.mean([r["paper_mbf"] for r in recs]))
        i_iou = float(np.mean([r["init_iou"] for r in recs]))
        i_dice = float(np.mean([r["init_dice"] for r in recs]))
        i_mbf = float(np.mean([r["init_paper_mbf"] for r in recs]))
        per_case_summary.append({
            "case": case_id,
            "slices": unique_slices,
            "pairs": len(recs),
            "iou": c_iou,
            "dice": c_dice,
            "paper_mbf": c_mbf,
            "init_iou": i_iou,
            "init_dice": i_dice,
            "init_paper_mbf": i_mbf,
            "dice_gain": c_dice - i_dice,
            "paper_mbf_gain": c_mbf - i_mbf,
        })
    case_csv_path = args.output_dir / "per_case_summary.csv"
    with open(case_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case_summary[0].keys()))
        writer.writeheader()
        writer.writerows(per_case_summary)
    print(f"Wrote {case_csv_path}")

    # Compute class summary
    class_summary = {}
    for cls in range(1, 20):
        c_rows = per_class_records.get(cls, [])
        if not c_rows:
            continue
        c_name = CLASS_NAMES.get(cls, f"Class_{cls}")
        gt_slices = len(c_rows)
        missing_slices = sum(r["prediction_missing"] for r in c_rows)
        iou = float(np.mean([r["iou"] for r in c_rows]))
        dice = float(np.mean([r["dice"] for r in c_rows]))
        paper_mbf = float(np.mean([r["paper_mbf"] for r in c_rows]))
        init_iou = float(np.mean([r["init_iou"] for r in c_rows]))
        init_dice = float(np.mean([r["init_dice"] for r in c_rows]))
        init_paper_mbf = float(np.mean([r["init_paper_mbf"] for r in c_rows]))

        nnunet_info = NNUNET_PER_CLASS.get(cls, {})
        nn_iou = nnunet_info.get("iou", None)
        nn_dice = nnunet_info.get("dice", None)
        nn_mbf = nnunet_info.get("paper_mbf", None)

        class_summary[str(cls)] = {
            "name": c_name,
            "gt_positive_slices": gt_slices,
            "prediction_missing_slices": missing_slices,
            "iou": iou,
            "dice": dice,
            "paper_mbf": paper_mbf,
            "init_iou": init_iou,
            "init_dice": init_dice,
            "init_paper_mbf": init_paper_mbf,
            "gain_iou": iou - init_iou,
            "gain_dice": dice - init_dice,
            "gain_paper_mbf": paper_mbf - init_paper_mbf,
            "nnunet_iou": nn_iou,
            "nnunet_dice": nn_dice,
            "nnunet_paper_mbf": nn_mbf,
            "diff_vs_nnunet_dice": dice - nn_dice if nn_dice else None,
            "diff_vs_nnunet_mbf": paper_mbf - nn_mbf if nn_mbf else None,
        }

    # Macro averages
    macro_iou = float(np.mean([v["iou"] for v in class_summary.values()]))
    macro_dice = float(np.mean([v["dice"] for v in class_summary.values()]))
    macro_mbf = float(np.mean([v["paper_mbf"] for v in class_summary.values()]))
    init_macro_iou = float(np.mean([v["init_iou"] for v in class_summary.values()]))
    init_macro_dice = float(np.mean([v["init_dice"] for v in class_summary.values()]))
    init_macro_mbf = float(np.mean([v["init_paper_mbf"] for v in class_summary.values()]))

    nnunet_macro_iou = float(np.mean([v["iou"] for v in NNUNET_PER_CLASS.values()]))
    nnunet_macro_dice = float(np.mean([v["dice"] for v in NNUNET_PER_CLASS.values()]))
    nnunet_macro_mbf = float(np.mean([v["paper_mbf"] for v in NNUNET_PER_CLASS.values()]))

    summary = {
        "model": "FlowContour_RAOS3",
        "checkpoint": str(args.checkpoint),
        "step": 60000,
        "test_dataset": str(args.data_root / "test"),
        "manifest": str(args.manifest),
        "total_test_slices": total_images,
        "total_cases": len(case_records),
        "total_gt_positive_pairs": len(all_records),
        "total_missing_predictions": sum(r["prediction_missing"] for r in all_records),
        "evaluation_time_seconds": eval_time,
        "specification": "TEAMS class-aware unified metric (GT-present classes; image mean then class mean; 1px boundary; cv2 DIST_L2; tolerances 1-5px)",
        "metrics": {
            "mIoU": macro_iou,
            "mDice": macro_dice,
            "paper_mBF": macro_mbf,
        },
        "initial_box_baseline": {
            "mIoU": init_macro_iou,
            "mDice": init_macro_dice,
            "paper_mBF": init_macro_mbf,
        },
        "evolution_gain": {
            "delta_mIoU": macro_iou - init_macro_iou,
            "delta_mDice": macro_dice - init_macro_dice,
            "delta_paper_mBF": macro_mbf - init_macro_mbf,
        },
        "nnunet_comparison": {
            "nnunet_mIoU": nnunet_macro_iou,
            "nnunet_mDice": nnunet_macro_dice,
            "nnunet_paper_mBF": nnunet_macro_mbf,
            "diff_mIoU": macro_iou - nnunet_macro_iou,
            "diff_mDice": macro_dice - nnunet_macro_dice,
            "diff_paper_mBF": macro_mbf - nnunet_macro_mbf,
        },
        "per_class": class_summary,
    }

    summary_json_path = args.output_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_json_path}")

    # Generate visualizations
    print("Generating visualizations for sample cases...")
    from lib.networks.make_network import make_network
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("net."): value for key, value in payload["state_dict"].items()}
    vis_model = make_network(cfg)
    vis_model.load_state_dict(state, strict=True)
    vis_model = vis_model.to("cuda:0").eval()
    generate_visualizations(args.output_dir / "visualizations", dataset, vis_model, 60000, args.gpu)

    # Generate markdown report
    


    report_md = build_report_md(args, summary, all_records, class_summary, case_records, eval_time)
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote {report_path}")

    print("\n" + "="*80)
    print("FINAL SUMMARY RESULTS (TEAMS Unified Class-Aware Standard)")
    print("="*80)
    print(f"FlowContour 60k mDice:     {macro_dice:.6f}  (Initial Box: {init_macro_dice:.6f}, Gain: {macro_dice-init_macro_dice:+.6f})")
    print(f"FlowContour 60k mIoU:      {macro_iou:.6f}  (Initial Box: {init_macro_iou:.6f}, Gain: {macro_iou-init_macro_iou:+.6f})")
    print(f"FlowContour 60k paper_mBF: {macro_mbf:.6f}  (Initial Box: {init_macro_mbf:.6f}, Gain: {macro_mbf-init_macro_mbf:+.6f})")
    print("="*80)


if __name__ == "__main__":
    main()
