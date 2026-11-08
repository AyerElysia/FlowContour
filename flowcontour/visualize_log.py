#!/usr/bin/env python3
"""Render the current RAOS3 training log once; rerun manually for a new image."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    ap.add_argument("--log", type=Path, default=root / "runs/pretrain/logs.jsonl")
    ap.add_argument("--output", type=Path, default=root / "runs/visualization/pretrain_log.png")
    ap.add_argument("--smooth", type=int, default=100,
                    help="rolling median window in effective updates")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = []
    for line in args.log.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # The training writer can be appending its final line.
        if all(k in row for k in ("step", "loss", "grad_norm", "lr")):
            records.append(row)
    if not records:
        raise ValueError(f"No complete update records in {args.log}")

    x = np.asarray([r["step"] for r in records])

    def values(key):
        return np.asarray([float(r.get(key, np.nan)) for r in records])

    def rolling_median(y):
        window = max(1, args.smooth)
        return np.asarray([np.nanmedian(y[max(0, i - window + 1):i + 1])
                           for i in range(len(y))])

    def plot_series(ax, key, label, color, log_scale=False):
        y = values(key)
        plot_x = x
        if key.startswith("rollout_"):
            keep = (x >= 1000) & (x % 4 == 0)
            plot_x, y = x[keep], y[keep]
        if not np.isfinite(y).any():
            return
        if log_scale:
            y = np.maximum(y, 1e-5)
            ax.set_yscale("log")
        ax.plot(plot_x, y, color=color, lw=0.55, alpha=0.34)
        ax.plot(plot_x, rolling_median(y), color=color, lw=1.45, label=label)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                             layout="constrained")
    plot_series(axes[0], "loss", "total loss", "#124c79", log_scale=True)
    plot_series(axes[0], "fm", "flow-matching", "#cf6a32", log_scale=True)
    plot_series(axes[0], "endpoint", "endpoint", "#4b8d4a", log_scale=True)
    plot_series(axes[0], "chamfer", "Chamfer", "#915da5", log_scale=True)
    axes[0].set_ylabel("Supervision")
    axes[0].legend(loc="upper right")

    plot_series(axes[1], "boundary", "boundary", "#38814a", log_scale=True)
    plot_series(axes[1], "geometry", "ordered geometry", "#8b5aa8", log_scale=True)
    plot_series(axes[1], "rollout_boundary", "rollout boundary", "#007c91", log_scale=True)
    plot_series(axes[1], "rollout_geometry", "rollout geometry", "#b06c00", log_scale=True)
    axes[1].set_ylabel("Auxiliary loss")
    axes[1].legend(loc="upper right", ncol=2)

    plot_series(axes[2], "grad_norm", "gradient norm", "#9b2d30", log_scale=True)
    axes[2].set_ylabel("Gradient norm")
    axes[2].legend(loc="upper right")

    axes[3].plot(x, values("lr"), color="#333333", lw=1.1)
    axes[3].set_ylabel("Learning rate")
    axes[3].set_xlabel("Effective update step")

    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle(f"FlowContour RAOS3 pretrain | latest step {int(x[-1])} | {len(records)} updates")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.stem + ".tmp" + args.output.suffix)
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    tmp.replace(args.output)  # Avoid a half-written image if the command is interrupted.
    print(f"Wrote {args.output} | latest_step={int(x[-1])} records={len(records)}")


if __name__ == "__main__":
    main()
