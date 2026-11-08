#!/usr/bin/env python3
"""One-shot RAOS case overlay from a completed FlowContour checkpoint."""
import argparse
import colorsys
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
NAMES = ('Liver','Spleen','Left kidney','Right kidney','Stomach','Gallbladder',
         'Esophagus','Pancreas','Duodenum','Colon','Intestine','Left adrenal',
         'Right adrenal','Rectum','Bladder','Left femur head','Right femur head',
         'Prostate','Seminal vesicles')

def color(label):
    rgb = colorsys.hsv_to_rgb(((label * 0.61803398875) % 1), 0.82, 1.0)
    return tuple(int(round(x * 255)) for x in rgb[::-1])

def latest_checkpoint():
    files = list((ROOT / 'weights').glob('*.pt'))
    if not files:
        raise FileNotFoundError('No pretrain checkpoints under runs/pretrain/checkpoints')
    return ROOT / 'weights/best.pt'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', type=Path, help='RAOS *_image.png path listed in configs/raos_manifest.csv; overrides case/slice selection')
    ap.add_argument('--case-id', default='case_0251')
    ap.add_argument('--slice-idx', type=int, help='Default: slice with most foreground classes, then area')
    ap.add_argument('--checkpoint', type=Path, help='Default: latest completed pretrain checkpoint')
    ap.add_argument('--output', type=Path, help='Default: runs/visualization/<case>_step_<N>.png')
    ap.add_argument('--seed', type=int, default=20260731)
    ap.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = ap.parse_args()
    os.chdir(ROOT)
    os.environ['FLOWCONTOUR_DATA_ROOT'] = str(DATA)
    os.environ['FLOWCONTOUR_SLICE_MANIFEST'] = str(ROOT / 'configs/raos_manifest.csv')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else os.environ.get('CUDA_VISIBLE_DEVICES','0')
    sys.argv = [sys.argv[0], '--cfg_file', str(ROOT / 'configs/pretrain.yaml')]

    import cv2
    import numpy as np
    import torch
    from lib.config import cfg
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import snake_collator
    from lib.networks import make_network

    torch.set_num_threads(2)
    ckpt = (args.checkpoint or latest_checkpoint()).resolve()
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    if payload.get('training_stage') not in ('pretrain','gdpo'):
        raise ValueError('Expected a pretrain or GDPO checkpoint')
    if args.image:
        image_path=args.image.resolve()
        try: split=image_path.relative_to(DATA.resolve()).parts[0]
        except ValueError as exc: raise ValueError(f'Image must be inside {DATA}') from exc
        dataset_name={'training':'RAOSTrain','validation':'RAOSVal','test':'RAOSTest'}.get(split)
        if not dataset_name: raise ValueError(f'Unsupported RAOS split: {split}')
    else:
        dataset_name='RAOSTest'
    dataset = make_dataset(cfg, dataset_name, is_train=False)
    if args.image:
        found=[(i,row) for i,row in enumerate(dataset.records) if Path(row['image_path']).resolve()==image_path]
        if not found: raise ValueError(f'Image is absent from the RAOS manifest: {image_path}')
        index,row=found[0]
        args.case_id=row['case_id']
    else:
        matches = [(i,row) for i,row in enumerate(dataset.records) if row['case_id'] == args.case_id]
        if not matches:
            raise ValueError(f'No validation slices for {args.case_id}')
    if args.image:
        pass
    elif args.slice_idx is None:
        def rank(item):
            mask = cv2.imread(item[1]['mask_path'], cv2.IMREAD_UNCHANGED)
            return (len(np.unique(mask[mask > 0])), int((mask > 0).sum()))
        index, row = max(matches, key=rank)
    else:
        found = [(i,row) for i,row in matches if row['slice_idx'] == args.slice_idx]
        if not found:
            raise ValueError(f'{args.case_id} has no slice {args.slice_idx}')
        index, row = found[0]
    # For a training-set image, use deterministic validation geometry for display.
    if dataset_name == 'RAOSTrain':
        dataset.split = 'val'
    sample = dataset[index]
    batch = snake_collator([sample])
    model = make_network(cfg)
    state = payload['state_dict']
    model.load_state_dict({k.removeprefix('net.'):v for k,v in state.items()}, strict=True)
    model = model.to(args.device).eval()
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        def move(value):
            if torch.is_tensor(value): return value.to('cuda')
            if isinstance(value,dict): return {k:move(v) for k,v in value.items()}
            if isinstance(value,list): return [move(v) for v in value]
            return value
        batch=move(batch)
    with torch.no_grad():
        result = model(batch['inp'], batch)
    pred = result['py'].detach().cpu().numpy()
    labels = np.asarray(sample['ct_cls'], dtype=np.int32)
    if len(pred) != len(labels):
        raise RuntimeError(f'Prediction/label instance count differs: {len(pred)} vs {len(labels)}')

    gray = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    shown = sorted(set(np.unique(mask).tolist()) - {0} | set(labels.tolist()))
    for label in shown:
        inside = mask == label
        if inside.any():
            tint = np.asarray(color(label), dtype=np.float32)
            canvas[inside] = np.clip(canvas[inside].astype(np.float32)*0.58 + tint*0.42,0,255).astype(np.uint8)
    inv = np.asarray(sample['meta']['inv_trans_input'], dtype=np.float32)
    for poly,label in zip(pred, labels):
        xy = poly * 4.0
        xy = xy @ inv[:,:2].T + inv[:,2]
        points = np.rint(xy).astype(np.int32).reshape(-1,1,2)
        cv2.polylines(canvas,[points],True,color(int(label)),2,lineType=cv2.LINE_AA)

    legend = np.full((canvas.shape[0],245,3),248,dtype=np.uint8)
    cv2.putText(legend,f'{args.case_id} / {row["slice_idx"]}',(10,27),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
    cv2.putText(legend,f'checkpoint step {payload["step"]}',(10,51),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
    cv2.putText(legend,'GT: filled | Pred: solid',(10,77),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
    for j,label in enumerate(shown):
        y=109+j*24
        cv2.rectangle(legend,(12,y-10),(29,y+5),color(label),-1)
        cv2.putText(legend,f'{label}: {NAMES[label-1] if 1 <= label <= len(NAMES) else "?"}',(38,y+2),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1,cv2.LINE_AA)
    out = args.output or ROOT/'runs/visualization'/f'{args.case_id}_slice{row["slice_idx"]}_step{payload["step"]}.png'
    out.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(out),np.concatenate([canvas,legend],axis=1)):
        raise RuntimeError(f'Failed to write {out}')
    print(f'Wrote {out} | case={args.case_id} slice={row["slice_idx"]} checkpoint_step={payload["step"]} contours={len(pred)}')

if __name__ == '__main__':
    main()
