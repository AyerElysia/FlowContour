#!/usr/bin/env python3
"""Select a completed RAOS3 GDPO checkpoint on the fixed validation panel."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-output', type=Path, required=True,
                        help='Completed formal GDPO run directory')
    parser.add_argument('--destination', type=Path, required=True,
                        help='New path for the selected GDPO checkpoint')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    train = args.train_output.resolve()
    destination = args.destination.resolve()
    if args.wait:
        while not (train / 'RESULT.json').is_file():
            state = json.loads((train / 'manifest.json').read_text())
            if state.get('status') == 'FAILED':
                (train / 'SELECTION_STATUS.json').write_text(json.dumps({
                    'status': 'TRAINING_FAILED', 'details': state.get('error')}, indent=2))
                return
            time.sleep(120)
    result = json.loads((train / 'RESULT.json').read_text())
    if result.get('status') != 'PASS' or result.get('last_update') != 3000:
        raise ValueError('Formal GDPO run has not completed 3000 updates')
    baseline = json.loads((train / 'validation_baseline.json').read_text())
    scored = []
    for path in sorted(train.glob('validation_step_*.json')):
        step = int(path.stem.rsplit('_', 1)[1])
        metrics = json.loads(path.read_text())
        checkpoint = train / 'checkpoints' / f'step_{step}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        scored.append({'step': step, 'quality': metrics['quality'],
                       'dice': metrics['dice'], 'iou': metrics['iou'],
                       'checkpoint': str(checkpoint)})
    if [r['step'] for r in scored] != [1000, 2000, 3000]:
        raise ValueError('Expected validation at 1000, 2000 and 3000')
    eligible = [r for r in scored if r['dice'] >= baseline['dice'] - 0.003]
    choice = max(eligible, key=lambda r: (r['quality'], r['dice'])) if eligible else None
    baseline_score = baseline['quality']
    if choice is None or choice['quality'] <= baseline_score:
        summary = {'status': 'NO_VALIDATED_RL_IMPROVEMENT', 'baseline': {
            'quality': baseline_score, 'dice': baseline['dice']},
            'candidates': scored, 'selected': None}
    else:
        source = Path(choice['checkpoint'])
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        summary = {'status': 'SELECTED', 'baseline': {
            'quality': baseline_score, 'dice': baseline['dice']},
            'candidates': scored, 'selected': choice,
            'best_gdpo_sha256': sha(destination)}
    (train / 'SELECTION_STATUS.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
