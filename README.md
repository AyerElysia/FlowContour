# FlowContour v2: RAOS code release

FlowContour combines YOLO26x visual features with a conditional flow model for
ordered medical contour evolution. The contour model uses a six-layer DiT,
contour-routed mixture of experts (CR-MoE), and geometric deformation policy
optimization (GDPO).

**This repository contains code only.** Model weights, medical images, masks,
detector labels, case/slice mappings, dataset manifests, and experiment result
files are not distributed here. The original release directory and its assets
are not modified by this publication. The previous repository version remains
available in Git history.

## Repository structure

```text
detector/
  models/                  YOLO26x architecture and model configuration
  build_raos_yolo_dataset.py
  train_raos_resized.py
  eval_raos.py
  prepare_yolo_test.py
  requirements.txt
flowcontour/
  configs/                 Shared, pretraining and GDPO configurations
  lib/                     Model, geometry, dataset, reward and training code
  tools/                   Training, checkpoint selection and evaluation
  visualize_case.py
  visualize_log.py
  requirements.txt
```

The training sequence in this v2 code is supervised flow-matching pretraining
followed by GDPO; there is no separate online imitation-learning stage in this
release. Contour inference uses two outer stages with four AB2 velocity
evaluations per stage. Consult the configuration and implementation for the
actual stage schedule, losses, and reward definition.

## Environment

Use Python 3.11. The supplied dependency lists are:

- `flowcontour/requirements.txt` for the contour implementation;
- `detector/requirements.txt` for the detector utilities.

For the contour environment:

```bash
python -m pip install -r flowcontour/requirements.txt
```

Run contour commands from `flowcontour/` so repository-relative configuration
paths resolve correctly. If using a separate detector environment, install its
requirements there; preserve the pinned contour dependencies when sharing one
environment.

## External data and checkpoints

Obtain a dataset and compatible model checkpoints separately, with the required
permissions. No download links or model files are included in this code release.

The dataset manifest must have the following columns:

```text
split,case_id,slice_idx,image_path,mask_path
```

`flowcontour/configs/raos_manifest.example.csv` contains only this header, not
patient records or dataset paths. Supply the real manifest through the relevant
command-line option. Utilities that use the default manifest location expect
your local file at `flowcontour/configs/raos_manifest.csv`; it is ignored by Git.
Use paths that resolve in the command's working directory; absolute image and
mask paths avoid ambiguity across the detector and contour utilities.

Model construction reads `yolo26_pretrained_path` in
`flowcontour/configs/common.yaml`, whose default is `weights/best.pt`. Provide
the appropriate checkpoint locally or edit that configuration to point to a
compatible detector checkpoint. Contour inference additionally requires a
compatible FlowContour checkpoint supplied with `--checkpoint`.

`lib/packaged_weights.py` supports a compatible combined checkpoint containing
embedded detector bytes, or a separately supplied Ultralytics detector
checkpoint. This loader support does **not** mean any weights are bundled with
this repository. Only load checkpoints from a trusted source.

Training requires the complete training/validation data and corresponding
manifest, not a test-only manifest. Some dataset preparation utilities enforce
the source release's expected test-set size; inspect their help and validation
checks before using a different split.

## Training entry points

```bash
cd flowcontour
python tools/train.py --help
python tools/start_gdpo.py --help
python tools/select_best_gdpo.py --help
```

`tools/train.py` provides supervised pretraining and GDPO modes, with explicit
preflight, checkpoint identity, output-directory and GPU checks. GDPO must
initialize from a compatible **pretraining** checkpoint, rather than from an
already post-trained inference checkpoint. For example, after preparing the
external assets:

```bash
python tools/start_gdpo.py \
  --checkpoint /path/to/pretrain_step_60000.pt \
  --data-root /path/to/full_raos_data \
  --slice-manifest /path/to/full_raos_manifest.csv \
  --output runs/gdpo --gpu 0
```

Use a new output directory. Checkpoint selection is validation-based; this
code-only publication does not include selected weights or experiment reports.

## Evaluation and visualization

With compatible external checkpoints and data configured:

```bash
cd flowcontour
python tools/extract_yolo_detections.py \
  --weights /path/to/detector_or_combined_checkpoint.pt \
  --manifest /path/to/test_manifest.csv \
  --output outputs/yolo_detections.json \
  --device 0 --conf 0.25 --iou 0.45

python tools/evaluate_yolo_plus_flowcontour.py \
  --checkpoint /path/to/flowcontour_checkpoint.pt \
  --detections outputs/yolo_detections.json \
  --data-root /path/to/raos_data \
  --manifest /path/to/test_manifest.csv \
  --output-dir outputs/end_to_end --gpu 0
```

`visualize_case.py` is an oracle-prompt visualization: it uses GT boxes and
classes, not the end-to-end detector pipeline. Its default local manifest and
data layout must be prepared before running it. Entry-point help is available
without providing model weights:

```bash
python visualize_case.py --help
python tools/evaluate_full_test.py --help
python tools/evaluate_yolo_plus_flowcontour.py --help
```

Oracle-prompt contour results and detector-driven end-to-end results must be
reported separately. This publication contains no new training or segmentation
accuracy claim.

## Core implementation

- `flowcontour/lib/networks/diffusion/flow_matching_evolution.py`: conditional
  flow actions and AB2 contour evolution.
- `flowcontour/lib/networks/diffusion/mainline_denoiser.py`: DiT denoising
  backbone and visual conditioning.
- `flowcontour/lib/networks/diffusion/cr_moe.py`: whole-contour expert routing.
- `flowcontour/lib/networks/diffusion/ordered_geometry.py`: ordered polygon
  geometry.
- `flowcontour/lib/networks/pure2d_yolo26_context.py`: YOLO26 P2/P3 features.
- `flowcontour/lib/rl/geometry.py`: geometric exploration.
- `flowcontour/lib/train/engine.py`: supervised training and GDPO updates.
- `flowcontour/lib/train/rewards/region_reward.py`: reward computation.

Generated outputs, real manifests, medical data and weight formats are ignored
by Git. Code publication does not grant redistribution rights for any external
dataset or model.
