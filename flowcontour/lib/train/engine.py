"""预训练、在线模仿、GDPO 共用的训练与检查点引擎。

run() 负责身份/协议闸门、冻结策略、有效更新循环和产物。
training_context() 只编码视觉上下文，不走策略梯度。
gdpo_step() 用几何探索候选更新策略，禁止走监督 FM forward。
"""

import copy
import hashlib
import json
import math
import random
import traceback
from pathlib import Path

import numpy as np
import torch

from lib.checkpoints import extract_state_dict, normalize_state_dict, sha256_file
from lib.config import cfg
from lib.datasets import make_data_loader
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.networks import make_network
from lib.protocol import protocol_metadata, PROTOCOL_VERSION
from lib.rl.geometry import sample_geometry, chart_logprob, chart_kl
from lib.train.rewards.region_reward import contour_metrics
from lib.train.states import terminal_credit
from lib.train.wrapper import FlowContourNetworkWrapper

# 签名 Dense 源：仅允许从该 SHA、step 19000 做 weights-only CR-MoE 迁移。
DENSE_SHA = "a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f"
# Dense 检查点里本来没有的张量：共享 router + DiT 第 2/4/6 层残差专家。
MISSING_PREFIXES = ("net.gcn.denoiser._contour_moe_router.",) + tuple(
    f"net.gcn.denoiser.dit_layers.{index}.routed_moe." for index in (1, 3, 5))
VAL_SHA = "24a4f19651edb5d187029f0255e2b59f9dce40f320ee29c14b709e8e92e6e6ad"


def dump_json(path, value):
    """原子写 JSON，禁止 NaN，避免半写入的 manifest。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_save(value, path):
    """原子写 torch 文件。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def move(value, device):
    """递归把 batch 树搬到设备。"""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value


def rng_state():
    """保存 python/numpy/torch/cuda RNG，供严格 resume。"""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    """按检查点逐项恢复 RNG。"""
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def tree_equal(a, b):
    """结构化精确相等，用于 optimizer/scheduler/RNG 回放校验。"""
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def code_identity(root):
    """代码+配置+位移统计的内容哈希，写入 identity，防止训练中途改代码。"""
    files = []
    for directory in ("lib", "tools", "configs"):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".csv"))
    # Displacements use per-contour scale normalization, not dataset statistics.
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted(files)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def load_weights(wrapper, payload, dense=False):
    """严格加载。dense=True 时只允许 MISSING_PREFIXES 中的新张量缺失。"""
    state = normalize_state_dict(extract_state_dict(payload))
    expected = wrapper.state_dict()
    if state and not next(iter(state)).startswith("net."):
        state = {"net." + key: tensor for key, tensor in state.items()}
    unexpected = set(state) - set(expected)
    missing = set(expected) - set(state)
    allowed = {key for key in expected if key.startswith(MISSING_PREFIXES)} if dense else set()
    wrong = [key for key in set(state) & set(expected) if state[key].shape != expected[key].shape]
    if unexpected or missing != allowed or wrong:
        raise RuntimeError(f"strict source mismatch: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}, shapes={wrong[:8]}")
    wrapper.load_state_dict(state, strict=not dense)
    for key, value in state.items():
        if not torch.equal(wrapper.state_dict()[key].cpu(), value.cpu()):
            raise RuntimeError(f"source tensor was not exactly restored: {key}")
    return {"loaded_tensors": len(state), "new_tensor_names": sorted(missing)}


def training_context(net, batch, limit=0):
    """Encode frozen YOLO/FPN context and preserve per-contour class identity."""
    with torch.no_grad():
        encoded = net.encode_batch(batch["inp"], batch)
        feature = encoded["cnn_feature"].detach()
        prepared = net.gcn.prepare_training(encoded, feature, batch)
        initial, target, owner = (prepared[key] for key in ("i_it_py", "i_gt_py", "py_ind"))
        class_ids = prepared["class_ids"]
        if limit:
            initial, target, owner, class_ids = initial[:limit], target[:limit], owner[:limit], class_ids[:limit]
    return feature, initial, target, owner, class_ids


def score(poly, target, batch):
    """网络输入分辨率上的 NSD/Dice/IoU。GDPO 只用 quality。"""
    return contour_metrics(poly, target, *batch["inp"].shape[-2:])


def validation_panel(root, smoke=False):
    """固定 Val37 子集。manifest SHA 变了就失败。smoke 只取 2 张。"""
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import snake_collator
    manifest = root / "configs/raos_manifest.csv"
    attrs = DatasetCatalog.get("RAOSVal")
    attrs["ann_file"] = str(manifest)
    DatasetCatalog.dataset_attrs["RAOSValidationPanel"] = attrs
    keys = ("sagittal_eval_include_case_ids", "sagittal_expected_eval_case_count", "sagittal_expected_eval_row_count")
    saved = [copy.deepcopy(cfg[key]) for key in keys]
    cfg.sagittal_eval_include_case_ids = []
    cfg.sagittal_expected_eval_case_count = 0
    cfg.sagittal_expected_eval_row_count = 0
    try:
        dataset = make_dataset(cfg, "RAOSValidationPanel", is_train=False)
    finally:
        for key, value in zip(keys, saved):
            cfg[key] = value
    if not dataset.records:
        raise RuntimeError("empty RAOS validation cohort")
    indices = np.linspace(0, len(dataset) - 1, min(80, len(dataset)), dtype=np.int64).tolist()
    selected = indices[:2] if smoke else indices
    panel = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selected),
                batch_size=1, num_workers=0, collate_fn=snake_collator)
    return panel, selected


@torch.no_grad()
def evaluate_panel(net, panel, device, seed):
    """生产推理路径评估。临时改 RNG，返回前恢复，避免污染训练流。"""
    previous_rng = rng_state()
    net.eval()
    rows, empty = [], 0
    try:
        for index, batch in enumerate(panel):
            torch.manual_seed(seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + index)
            batch = move(batch, device)
            feature, initial, target, owner, class_ids = training_context(net, batch)
            if not initial.numel():
                empty += 1
                continue
            predicted = net(batch["inp"], batch)
            if predicted["py"].shape != target.shape:
                raise RuntimeError("production inference/GT instance shape mismatch")
            metrics = score(predicted["py"], target, batch)
            initial_metrics = score(initial, target, batch)
            metrics["initial_dice"] = initial_metrics["dice"]
            metrics["dice_gain_over_init"] = metrics["dice"] - initial_metrics["dice"]
            metrics["motion_input_px"] = (predicted["py"] - initial).norm(dim=-1).mean(1) * 4
            rows.append({key: float(value.mean()) for key, value in metrics.items()})
        return {"rows": len(rows), "empty_gt_rows": empty,
                **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]}} if rows else {"rows": 0, "empty_gt_rows": empty}
    finally:
        restore_rng(previous_rng)
        net.train()


def gdpo_step(net, reference, batch, optimizer, debug):
    """一条 batch 上的 GDPO：K=8 几何候选 × 两步完整动作，再用终点信用做 PPO。

    两步共享同一组 latent，与部署采样一致。几何分量在采样时冻结，replay 用 chart_logprob。
    KL 相对冻结的原始 source reference，不是 resume 后的当前策略。
    某 epoch 近似 KL 过大则丢掉该次 pending step；若整步没有任何 optimizer.step，则失败。
    """
    feature, initial, target, owner, class_ids = training_context(net, batch, cfg.gdpo.max_contours)
    if not initial.numel():
        return None
    k = int(cfg.gdpo.k)
    if k != 8 or list(cfg.gdpo.local_quota) not in ([0, 1], [0, 0]):
        raise ValueError("supported GDPO profiles: K8 mixed [0,1], matched low-only [0,0]")
    from lib.networks.diffusion.ordered_geometry import correlated_noise
    latent = [correlated_noise(initial) for _ in range(2)]
    local_indices = [set(torch.randperm(k)[:int(quota)].tolist()) for quota in cfg.gdpo.local_quota]
    trajectories, old_logs, components = [], [], []
    with torch.no_grad():
        for candidate in range(k):
            records, logs = [], []

            def sampler(stage, state, mean):
                # sigma_px 是输入像素，除以 down_ratio=4 才是特征网格。
                debug.update(stage=stage, candidate=candidate, state=state.cpu(),
                             mean=mean.cpu(), latent=latent[stage].cpu())
                action, logprob, component = sample_geometry(
                    mean, state, cfg.gdpo.sigma_px[stage] / 4,
                    local=candidate in local_indices[stage], modes=cfg.gdpo.modes)
                records.append(component)
                logs.append(logprob)
                return action

            trajectories.append(net.gcn.rollout(feature, initial, owner, latent, sampler,
                                               class_ids=class_ids))
            components.append(records)
            old_logs.append(logs)
        # polys[1:] 是两步结束后的轮廓；quality[:, -1] 才是终点奖励。
        quality = torch.stack([torch.stack([score(poly, target, batch)["quality"]
                              for poly in row["polys"][1:]]) for row in trajectories])
        advantage, weights, allocated = terminal_credit(
            quality[:, -1], quality, cfg.gdpo.adv_std_floor, cfg.gdpo.adv_clip)
    losses, kls, replay_errors = [], [], []
    gradient = 0.
    applied_updates = 0
    for epoch in range(cfg.gdpo.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_kls = []
        for candidate, row in enumerate(trajectories):
            for stage in range(2):
                state = row["polys"][stage]
                action = row["actions"][stage]
                component = components[candidate][stage]
                mean = net.gcn.action_mean(feature, state, owner, stage, latent[stage],
                                          class_ids=class_ids)
                logprob = chart_logprob(action, mean, component)
                difference = logprob - old_logs[candidate][stage]
                if epoch == 0:
                    replay_errors.append(float(difference.detach().abs().max()))
                    if replay_errors[-1] >= .002:
                        raise RuntimeError("unchanged-policy probability replay exceeded .002")
                ratio = difference.exp()
                if not torch.isfinite(ratio).all():
                    raise FloatingPointError("nonfinite policy ratio")
                approx_kl = ((ratio - 1) - difference).mean()
                with torch.no_grad():
                    reference_mean = reference.action_mean(feature, state, owner, stage, latent[stage],
                                                          class_ids=class_ids)
                regularization = chart_kl(mean, reference_mean, component).mean()
                policy = -torch.minimum(ratio * allocated[candidate, stage],
                         ratio.clamp(1 - cfg.gdpo.clip, 1 + cfg.gdpo.clip) * allocated[candidate, stage]).mean()
                loss = (policy + cfg.gdpo.kl_beta * regularization) / (2 * k)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite GDPO loss")
                loss.backward()
                losses.append(float(loss.detach()))
                epoch_kls.append(float(approx_kl.detach()))
        # A KL stop discards the pending oversized step, not the entire run.
        kls.extend(epoch_kls)
        if sum(epoch_kls) / len(epoch_kls) > cfg.gdpo.approx_kl_stop:
            optimizer.zero_grad(set_to_none=True)
            break
        gradient = float(torch.nn.utils.clip_grad_norm_(net.gcn.parameters(), cfg.gdpo.grad_clip,
                                                       error_if_nonfinite=True))
        optimizer.step()
        applied_updates += 1
    debug.update(quality=quality.cpu(), advantages=allocated.cpu(), weights=weights.cpu(),
                 trajectories=move(trajectories, "cpu"), components=move(components, "cpu"),
                 target=target.cpu(), image=batch["inp"].cpu())
    if not applied_updates:
        raise RuntimeError("no optimizer update applied; cannot count an effective step")
    return {"loss": sum(losses), "reward": float(quality[:, -1].mean()),
            "candidate_reward_std": float(quality[:, -1].std(0, unbiased=False).mean()),
            "grad_norm": gradient, "approx_kl": sum(kls) / max(len(kls), 1),
            "probability_replay_error": max(replay_errors),
            "credit_conservation_error": float((allocated.mean(1) - advantage).abs().max()),
            "credit_sign_violations": int((allocated * advantage[:, None] < 0).sum()),
            "local_candidates_by_stage": [len(value) for value in local_indices],
            "outer_actions_per_rollout": 2}


def run(args, gpu_checks):
    """完整一次实验：校验源权重与协议，按阶段冻结，循环有效更新，写出检查点与 RESULT。"""
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    code_hash, code_files = code_identity(root)
    source = Path(args.source_checkpoint).resolve() if args.source_checkpoint else None
    if not args.random_init and sha256_file(source) != args.source_sha256:
        raise RuntimeError("source SHA256 mismatch")
    payload = {} if args.random_init else torch.load(source, map_location="cpu", weights_only=False)
    if not args.random_init and int(payload.get("step", -1)) != args.source_step:
        raise RuntimeError("source step mismatch")
    stage = cfg.training_stage
    # Dense 迁移是唯一允许没有 v2 protocol 的初始化路径。
    if args.initialize_dense:
        if stage != "pretrain" or args.source_sha256 != DENSE_SHA or args.source_step != 19000:
            raise RuntimeError("Dense migration requires the signed 19000 source and pretrain stage")
    elif not args.random_init and payload.get("protocol") != protocol_metadata():
        raise RuntimeError("source protocol mismatch: initialize legacy weights ONLY via signed Dense pretraining")
    if stage == "gdpo" and payload.get("training_stage") != "pretrain":
        raise RuntimeError("posttraining source must come from the preceding training stage")
    if args.mode == "train" and not (args.initialize_dense or args.random_init) and payload.get("run_mode") != "train":
        raise RuntimeError("a smoke/preflight checkpoint cannot initialize formal training")
    settings = cfg.dump()
    identity = {"code": code_hash, "config": hashlib.sha256(settings.encode()).hexdigest(),
                "source": args.source_sha256 or "random_init", "stage": stage,
                "data_root": str(Path(args.data_root).resolve()),
                "manifest": sha256_file(args.slice_manifest),
                "yolo26_checkpoint": sha256_file(cfg.yolo26_pretrained_path), "seed": args.seed,
                "device": args.device, "shared_gpu": bool(args.allow_shared_gpu)}
    if args.mode == "train":
        preflight = json.loads((Path(args.preflight_output) / "RESULT.json").read_text())
        if preflight["status"] != "PASS" or preflight["identity"] != identity or preflight["mode"] != "preflight" or preflight["functional_smoke"]:
            raise RuntimeError("formal run requires a matching completed preflight")
    functional_smoke = args.mode == "preflight" and args.device == "cpu"
    runtime_batch = 1 if functional_smoke else int(cfg.train.batch_size)
    runtime_workers = 0 if args.mode == "preflight" else int(cfg.train.num_workers)
    manifest = {"status": "RUNNING", "identity": identity, "protocol": protocol_metadata(),
                "mode": args.mode, "gpu_checks": gpu_checks,
                "shared_gpu": args.allow_shared_gpu, "functional_smoke": functional_smoke,
                "runtime_batch_size": runtime_batch, "runtime_workers": runtime_workers,
                "reportable": False, "code_files": code_files}
    dump_json(output / "manifest.json", manifest)
    (output / "config.yaml").write_text(settings, encoding="utf-8")
    wrapper = FlowContourNetworkWrapper(make_network(cfg)).to(args.device)
    if sum(p.numel() for p in wrapper.parameters()) != int(cfg.pure2d_expected_parameter_count):
        raise RuntimeError("model parameter contract changed")
    source_audit = {"random_init": True} if args.random_init else load_weights(wrapper, payload, args.initialize_dense)
    del payload
    net = wrapper.net
    # Pretrain FPN + Flow; GDPO freezes YOLO/FPN and trains the contour policy.
    for parameter in net.parameters():
        parameter.requires_grad_(stage == "pretrain")
    for parameter in net.gcn.parameters():
        parameter.requires_grad_(True)
    # Preserve the tuned RAOS YOLO26 backbone; only P2/P3 FPN trains in pretrain.
    net.locate_feat_replacer.backbone.requires_grad_(False)
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    target_steps = 2 if args.mode == "preflight" else int(cfg.train.max_steps)

    def lr_factor(step):
        """仅预训练做 warmup + cosine；online/GDPO 保持恒定 LR。"""
        if stage != "pretrain":
            return 1.
        warmup = int(cfg.train.warmup_steps)
        if step < warmup:
            return max(step + 1, 1) / max(warmup, 1)
        progress = (step - warmup) / max(cfg.train.max_steps - warmup, 1)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    # Freeze the ORIGINAL source reference, never the resumed policy.
    reference = copy.deepcopy(net.gcn).eval().requires_grad_(False) if stage == "gdpo" else None
    frozen_before = {key: value.detach().cpu().clone() for key, value in wrapper.state_dict().items()
                     if stage != "pretrain" and not key.startswith("net.gcn.")}
    start_step = 0
    resume_audit = None
    if args.resume_checkpoint:
        resumed = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if resumed["identity"] != identity or resumed["protocol"] != protocol_metadata():
            raise RuntimeError("strict resume identity/protocol mismatch")
        load_weights(wrapper, resumed)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        restore_rng(resumed["rng"])
        if not tree_equal(optimizer.state_dict(), resumed["optimizer"]):
            raise RuntimeError("optimizer restoration differs from checkpoint")
        if not tree_equal(scheduler.state_dict(), resumed["scheduler"]):
            raise RuntimeError("scheduler restoration differs from checkpoint")
        if not tree_equal(rng_state(), resumed["rng"]):
            raise RuntimeError("RNG restoration differs from checkpoint")
        start_step = int(resumed["step"])
        if not 0 <= start_step < target_steps:
            raise ValueError("resume step must be below this run's target")
        resume_audit = {"step": start_step, "model": True, "optimizer": True,
                        "scheduler": True, "rng": True, "data_stream_bitwise_replay": False}
        del resumed
    if any(not torch.equal(wrapper.state_dict()[key].detach().cpu(), value)
           for key, value in frozen_before.items()):
        raise RuntimeError("resume modified tensors frozen relative to the original source")
    before = {key: value.detach().cpu().clone() for key, value in wrapper.state_dict().items()}
    cfg.train.batch_size = runtime_batch
    cfg.train.num_workers = runtime_workers
    loader = make_data_loader(cfg, is_train=True)
    panel, panel_indices = validation_panel(root, functional_smoke)
    baseline = evaluate_panel(net, panel, args.device, args.seed + 91000000)
    dump_json(output / "validation_baseline.json", {"indices": panel_indices, "reportable": False, **baseline})
    if stage == "gdpo" and not functional_smoke and (baseline.get("iou", 0) < cfg.gdpo.min_baseline_iou or baseline.get("dice", 0) < cfg.gdpo.min_baseline_dice):
        raise RuntimeError(f"GDPO source baseline sanity gate failed: {baseline}")
    iterator = iter(loader)
    step, empty_streak = start_step, 0
    debug = {}
    try:
        while step < target_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move(batch, args.device)
            if not batch["ct_01"].bool().any():
                empty_streak += 1
                if empty_streak > max(len(loader), 32):
                    raise RuntimeError("too many empty batches")
                continue
            empty_streak = 0
            net.train()
            if stage != "pretrain":
                net.locate_feat_replacer.eval()
            optimizer.zero_grad(set_to_none=True)
            if stage == "gdpo":
                metrics = gdpo_step(net, reference, batch, optimizer, debug)
                if metrics is None:
                    continue
            else:
                cfg.train.current_update=step+1
                prediction = net(batch["inp"], batch)
                from lib.train.boundary_objective import boundary_loss, geometry_losses
                gt = prediction["i_gt_py"]
                dense_gt = batch["dense_gt"][batch["ct_01"].bool()]
                dense_weights = batch["dense_weights"][batch["ct_01"].bool()]
                boundary = boundary_loss(prediction["py"], gt, dense_gt, dense_weights)
                ordered, _ = geometry_losses(prediction["py"], gt)
                ramp = min((step+1) / max(int(cfg.train.geometry_warmup_steps), 1), 1.0)
                loss = (prediction["diff_loss"] +
                        ramp * (float(cfg.train.boundary_weight)*boundary +
                                float(cfg.train.geometry_weight)*ordered))
                rollout_boundary=loss.new_zeros(())
                rollout_geometry=loss.new_zeros(())
                if (args.mode=="train" and step+1>=int(cfg.train.geometry_warmup_steps)
                        and (step+1)%int(cfg.train.rollout_every)==0):
                    rolled=net.gcn.rollout(prediction["cnn_feature"],
                                           prediction["i_it_py"],prediction["py_ind"],
                                           class_ids=prediction["class_ids"])["py"]
                    rollout_boundary=boundary_loss(rolled,gt,dense_gt,dense_weights)
                    rollout_geometry,_=geometry_losses(rolled,gt,align=True)
                    loss=(loss+float(cfg.train.rollout_boundary_weight)*rollout_boundary
                              +float(cfg.train.rollout_geometry_weight)*rollout_geometry)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite FM loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.gradient_clip,
                                                         error_if_nonfinite=True)
                optimizer.step()
                metrics = {"loss": float(loss.detach()), "grad_norm": float(gradient),
                           "fm": float(prediction["diff_loss_data"]),
                           "endpoint": float(prediction["diff_loss_endpoint"]),
                           "chamfer": float(prediction["diff_loss_chamfer"]),
                           "boundary": float(boundary.detach()),
                           "geometry": float(ordered.detach()),
                           "rollout_boundary": float(rollout_boundary.detach()),
                           "rollout_geometry": float(rollout_geometry.detach()),
                           "state_counts": prediction["state_counts"].tolist(),
                           "phase_mean": float(prediction["phase_mean"]),
                           "online_states_detached": False}
                del prediction
            scheduler.step()
            step += 1
            metrics.update(step=step, target_steps=target_steps, lr=optimizer.param_groups[0]["lr"])
            with (output / "logs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            print(json.dumps(metrics), flush=True)
            if step == target_steps or args.mode == "preflight" or step % cfg.train.step_checkpoint_every == 0:
                current = wrapper.state_dict()
                changed = [key for key, value in current.items() if not torch.equal(value.detach().cpu(), before[key])]
                frozen_changed = [key for key, value in frozen_before.items() if not torch.equal(current[key].detach().cpu(), value)]
                if frozen_changed:
                    raise RuntimeError(f"frozen tensors changed: {frozen_changed[:4]}")
                if any(t.is_floating_point() and not torch.isfinite(t).all() for t in current.values()):
                    raise FloatingPointError("nonfinite checkpoint")
                checkpoint = {"format_version": 3, "state_dict": current, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": rng_state(), "step": step,
                    "target_steps": target_steps, "training_stage": stage, "identity": identity,
                    "run_mode": args.mode,
                    "protocol": protocol_metadata(), "metrics": metrics}
                atomic_save(checkpoint, output / "checkpoints" / f"step_{step}.pt")
                if args.mode == "train" and step % cfg.gdpo.eval_every == 0:
                    validation = evaluate_panel(net, panel, args.device, args.seed + 91000000)
                    dump_json(output / f"validation_step_{step}.json", {"indices": panel_indices,
                              "reportable": False, **validation})
        if not any(key.startswith("net.gcn.") for key in changed):
            raise RuntimeError("no Flow tensors updated")
        if stage == "pretrain" and not any(key.startswith("net.locate_feat_replacer.") for key in changed):
            raise RuntimeError("no feature replacer tensors updated")
        # 用同一真实 batch 走生产推理，钩 denoiser 确认恰好 8 NFE。
        net.eval()
        calls = []
        hook = net.gcn.denoiser.register_forward_hook(lambda *values: calls.append(1))
        with torch.no_grad():
            predicted = net(batch["inp"], batch)
        hook.remove()
        if len(calls) != 8 or not torch.isfinite(predicted["py"]).all():
            raise RuntimeError("production inference NFE/finite check failed")
        inference = {"nfe": len(calls), "shape": list(predicted["py"].shape), "finite": True}
        if code_identity(root)[0] != code_hash:
            raise RuntimeError("training code changed during execution")
        result = {**manifest, "status": "PASS", "effective_updates": step - start_step,
                  "first_update": start_step + 1, "last_update": step,
                  "changed_flow_tensors": sum(key.startswith("net.gcn.") for key in changed),
                  "changed_replacer_tensors": sum(key.startswith("net.locate_feat_replacer.") for key in changed),
                  "frozen_changed_tensors": len(frozen_changed), "source_audit": source_audit,
                  "resume_audit": resume_audit, "inference_smoke": inference}
        result["validation_panel_indices"] = panel_indices
        result["baseline_sanity_enforced"] = stage == "gdpo" and not functional_smoke
        if debug:
            atomic_save(debug, output / "actual_training_candidates.pt")
            from lib.visualizers.contours import save_candidates
            save_candidates(output / "actual_training_candidates.png", debug["image"],
                            debug["target"], debug["trajectories"], debug["quality"])
        dump_json(output / "RESULT.json", result)
        dump_json(output / "manifest.json", result)
        return result
    except Exception as error:
        atomic_save({"debug": debug, "batch": move(batch, "cpu"), "state_dict": wrapper.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": rng_state()}, output / "failure_snapshot.pt")
        dump_json(output / "manifest.json", {**manifest, "status": "FAILED", "step": step,
                  "error": repr(error), "traceback": traceback.format_exc()})
        raise
