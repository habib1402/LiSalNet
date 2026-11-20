#!/usr/bin/env python3
"""Evaluate saved LiSalNet PNG maps with the full RGB-D SOD metric set."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib.metadata
import json

import cv2
import numpy as np
import py_sod_metrics
from tqdm import tqdm

from dataset import resolve_manifest_path


def evaluate_split(data_root, prediction_root, split):
    manifest = Path(data_root) / f'{split}.txt'
    rows = [line.split() for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or any(len(row) != 2 for row in rows):
        raise ValueError(f'{manifest}: expected nonempty image_path mask_path pairs.')
    stems = [Path(row[0]).stem for row in rows]
    if len(stems) != len(set(stems)):
        raise ValueError(f'{manifest}: duplicate image basenames are ambiguous.')

    fm = py_sod_metrics.FmeasureV2(metric_handlers={
        'fm': py_sod_metrics.FmeasureHandler(
            with_adaptive=True, with_dynamic=True, beta=0.3
        )
    })
    wfm = py_sod_metrics.WeightedFmeasure()
    sm = py_sod_metrics.Smeasure()
    em = py_sod_metrics.Emeasure()
    mae = py_sod_metrics.MAE()
    split_root = Path(prediction_root) / split

    for (image_entry, mask_entry), stem in tqdm(
            zip(rows, stems), total=len(rows), desc=split):
        del image_entry  # The image name determines the prediction name only.
        pred_path = split_root / f'{stem}.png'
        gt_path = resolve_manifest_path(data_root, mask_entry)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(f'Could not read prediction: {pred_path}')
        if gt is None:
            raise FileNotFoundError(f'Could not read ground truth: {gt_path}')
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        for metric in (fm, wfm, sm, em, mae):
            metric.step(pred=pred, gt=gt)

    fm_result = fm.get_results()['fm']
    em_result = em.get_results()['em']
    return {
        'images': len(rows),
        'MAE': float(mae.get_results()['mae']),
        'S_measure': float(sm.get_results()['sm']),
        'max_F_measure': float(np.max(fm_result['dynamic'])),
        'mean_F_measure': float(np.mean(fm_result['dynamic'])),
        'adaptive_F_measure': float(fm_result['adaptive']),
        'max_E_measure': float(np.max(em_result['curve'])),
        'mean_E_measure': float(np.mean(em_result['curve'])),
        'adaptive_E_measure': float(em_result['adp']),
        'weighted_F_measure': float(wfm.get_results()['wfm']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, type=Path)
    parser.add_argument('--pred_root', required=True, type=Path)
    parser.add_argument('--lists', nargs='+',
                        default=['NJU2K_test', 'NLPR_test', 'SSD', 'STERE', 'SIP', 'DES'])
    parser.add_argument('--output', type=Path, default=Path('evaluation_full.json'))
    args = parser.parse_args()
    results = {
        split: evaluate_split(args.data_dir, args.pred_root, split)
        for split in args.lists
    }
    report = {
        'evaluator': 'pysodmetrics',
        'evaluator_version': importlib.metadata.version('pysodmetrics'),
        'prediction_format': 'saved uint8 PNG maps',
        'results': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
