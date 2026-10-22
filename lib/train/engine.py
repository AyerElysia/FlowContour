"""Shared training/checkpoint engine for pretrain, online imitation and GDPO."""

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

DENSE_SHA = "a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f"
MISSING_PREFIXES = ("net.gcn.denoiser._contour_moe_router.",) + tuple(
    f"net.gcn.denoiser.dit_layers.{index}.routed_moe." for index in (1, 3, 5))
VAL_SHA = "24a4f19651edb5d187029f0255e2b59f9dce40f320ee29c14b709e8e92e6e6ad"


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def tree_equal(a, b):
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
    files = []
    for directory in ("lib", "tools", "configs"):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".csv"))
    files.append(root / "data/stats/volmem_sagittal_disp_stats.json")
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted(files)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def load_weights(wrapper, payload, dense=False):
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
    with torch.no_grad():
        encoded = net.encode_batch(batch["inp"], batch)
        feature = encoded["cnn_feature"].detach()
        prepared = net.gcn.prepare_training(encoded, feature, batch)
        initial, target, owner = (prepared[key] for key in ("i_it_py", "i_gt_py", "py_ind"))
        if limit:
            initial, target, owner = initial[:limit], target[:limit], owner[:limit]
    return feature, initial, target, owner


def score(poly, target, batch):
    return contour_metrics(poly, target, *batch["inp"].shape[-2:])


def validation_panel(root, smoke=False):
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import snake_collator
    manifest = root / "configs/manifests/volmem_fourier_validation37.csv"
    if sha256_file(manifest) != VAL_SHA:
        raise RuntimeError("fixed Val37 manifest changed")
    attrs = DatasetCatalog.get("VolMemVal")
    attrs["ann_file"] = str(manifest)
    DatasetCatalog.dataset_attrs["VolMemValidation37"] = attrs
    keys = ("sagittal_eval_include_case_ids", "sagittal_expected_eval_case_count", "sagittal_expected_eval_row_count")
    saved = [copy.deepcopy(cfg[key]) for key in keys]
    cfg.sagittal_eval_include_case_ids = []
    cfg.sagittal_expected_eval_case_count = 0
    cfg.sagittal_expected_eval_row_count = 0
    try:
        dataset = make_dataset(cfg, "VolMemValidation37", is_train=False)
    finally:
        for key, value in zip(keys, saved):
            cfg[key] = value
    if len(dataset) != 5914 or len({row["case_id"] for row in dataset.records}) != 37:
        raise RuntimeError("fixed Val37 cohort differs")
    indices = np.linspace(0, len(dataset) - 1, 80, dtype=np.int64).tolist()
    selected = indices[:2] if smoke else indices
    panel = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selected),
                batch_size=1, num_workers=0, collate_fn=snake_collator)
    return panel, selected


@torch.no_grad()
def evaluate_panel(net, panel, device, seed):
    previous_rng = rng_state()
    net.eval()
    rows, empty = [], 0
    try:
        for index, batch in enumerate(panel):
            torch.manual_seed(seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + index)
            batch = move(batch, device)
            feature, initial, target, owner = training_context(net, batch)
            if not initial.numel():
                empty += 1
                continue
            predicted = net(batch["inp"], batch)
            if predicted["py"].shape != target.shape:
                raise RuntimeError("production inference/GT instance shape mismatch")
            metrics = score(predicted["py"], target, batch)
            rows.append({key: float(value.mean()) for key, value in metrics.items()})
        return {"rows": len(rows), "empty_gt_rows": empty,
                **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]}} if rows else {"rows": 0, "empty_gt_rows": empty}
    finally:
        restore_rng(previous_rng)
        net.train()


def gdpo_step(net, reference, batch, optimizer, debug):
    feature, initial, target, owner = training_context(net, batch, cfg.gdpo.max_contours)
    if not initial.numel():
        return None
    k = int(cfg.gdpo.k)
    if k != 8 or list(cfg.gdpo.local_quota) not in ([0, 1], [0, 0]):
        raise ValueError("supported GDPO profiles: K8 mixed [0,1], matched low-only [0,0]")
    latent = [torch.randn_like(initial) for _ in range(2)]
    local_indices = [set(torch.randperm(k)[:int(quota)].tolist()) for quota in cfg.gdpo.local_quota]
    trajectories, old_logs, components = [], [], []
    with torch.no_grad():
        for candidate in range(k):
            records, logs = [], []

            def sampler(stage, state, mean):
                debug.update(stage=stage, candidate=candidate, state=state.cpu(),
                             mean=mean.cpu(), latent=latent[stage].cpu())
                action, logprob, component = sample_geometry(
                    mean, state, cfg.gdpo.sigma_px[stage] / 4,
                    local=candidate in local_indices[stage], modes=cfg.gdpo.modes)
                records.append(component)
                logs.append(logprob)
                return action

            trajectories.append(net.gcn.rollout(feature, initial, owner, latent, sampler))
            components.append(records)
            old_logs.append(logs)
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
                mean = net.gcn.action_mean(feature, state, owner, stage, latent[stage])
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
                    reference_mean = reference.action_mean(feature, state, owner, stage, latent[stage])
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
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    code_hash, code_files = code_identity(root)
    source = Path(args.source_checkpoint).resolve()
    if sha256_file(source) != args.source_sha256:
        raise RuntimeError("source SHA256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != args.source_step:
        raise RuntimeError("source step mismatch")
    stage = cfg.training_stage
    if args.initialize_dense:
        if stage != "pretrain" or args.source_sha256 != DENSE_SHA or args.source_step != 19000:
            raise RuntimeError("Dense migration requires the signed 19000 source and pretrain stage")
    elif payload.get("protocol") != protocol_metadata():
        raise RuntimeError("source protocol mismatch: initialize legacy weights ONLY via signed Dense pretraining")
    if stage in ("online", "gdpo") and payload.get("training_stage") != {"online": "pretrain", "gdpo": "online"}[stage]:
        raise RuntimeError("posttraining source must come from the preceding training stage")
    if args.mode == "train" and not args.initialize_dense and payload.get("run_mode") != "train":
        raise RuntimeError("a smoke/preflight checkpoint cannot initialize formal training")
    settings = cfg.dump()
    identity = {"code": code_hash, "config": hashlib.sha256(settings.encode()).hexdigest(),
                "source": args.source_sha256, "stage": stage,
                "data_root": str(Path(args.data_root).resolve()),
                "manifest": sha256_file(args.slice_manifest),
                "moonvit_cache": str(Path(args.moonvit_cache).resolve()), "seed": args.seed,
                "device": args.device}
    if args.mode == "train":
        preflight = json.loads((Path(args.preflight_output) / "RESULT.json").read_text())
        if preflight["status"] != "PASS" or preflight["identity"] != identity or preflight["mode"] != "preflight" or preflight["functional_smoke"]:
            raise RuntimeError("formal run requires a matching completed preflight")
    functional_smoke = args.mode == "preflight" and (args.allow_shared_gpu or args.device == "cpu")
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
    if sum(p.numel() for p in wrapper.parameters()) != 16933968:
        raise RuntimeError("model parameter contract changed")
    source_audit = load_weights(wrapper, payload, args.initialize_dense)
    del payload
    net = wrapper.net
    for parameter in net.parameters():
        parameter.requires_grad_(stage == "pretrain")
    for parameter in net.gcn.parameters():
        parameter.requires_grad_(True)
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    target_steps = 2 if args.mode == "preflight" else int(cfg.train.max_steps)

    def lr_factor(step):
        if stage != "pretrain":
            return 1.
        warmup = int(cfg.train.warmup_steps)
        if step < warmup:
            return max(step + 1, 1) / max(warmup, 1)
        progress = (step - warmup) / max(cfg.train.max_steps - warmup, 1)
        return .5 * (1 + math.cos(math.pi * min(progress, 1)))

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
                prediction = net(batch["inp"], batch)
                loss = prediction["diff_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite FM loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.gradient_clip,
                                                         error_if_nonfinite=True)
                optimizer.step()
                metrics = {"loss": float(loss.detach()), "grad_norm": float(gradient),
                           "state_counts": prediction["state_counts"].tolist(),
                           "phase_mean": float(prediction["phase_mean"]),
                           "online_states_detached": stage == "online"}
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
        # Exercise the production inference path with the same real batch.
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
