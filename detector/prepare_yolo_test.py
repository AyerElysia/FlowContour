#!/usr/bin/env python3
"""Create a flat YOLO test image view from the packaged RAOS test PNGs."""
import argparse
import csv
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'flowcontour/configs/raos_manifest.csv'
DEST = ROOT / 'data/yolo_test/images/test'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--copy', action='store_true', help='Copy instead of creating relative symlinks')
    args = ap.parse_args()
    rows = list(csv.DictReader(MANIFEST.open(newline='', encoding='utf-8')))
    if len(rows) != 2648 or {r['split'] for r in rows} != {'test'}:
        raise ValueError('Unexpected test manifest')
    DEST.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image = (ROOT / 'data' / row['image_path']).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        target = DEST / image.name
        if target.exists():
            if target.resolve() != image and not args.copy:
                raise FileExistsError(target)
            continue
        if args.copy:
            shutil.copy2(image, target)
        else:
            target.symlink_to(Path('../../../test') / row['case_id'] / image.name)
    print(f'Ready: {len(rows)} test images under {DEST}')


if __name__ == '__main__':
    main()
