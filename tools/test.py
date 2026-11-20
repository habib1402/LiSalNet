#!/usr/bin/env python3
"""Run LiSalNet P1 inference and save original-resolution saliency maps."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models import LiSalNet
from checkpoint_utils import load_checkpoint
from SalEval import SalEval
from dataset import resolve_manifest_path


@torch.inference_mode()
def evaluate(model, args, split, device):
    root = Path(args.data_dir)
    rows = [line.split() for line in (root / (split + '.txt')).read_text().splitlines() if line.strip()]
    if not rows or any(len(row) != 2 for row in rows):
        raise ValueError(f'{split}: expected nonempty image_path mask_path pairs.')
    names = [Path(row[0]).stem for row in rows]
    if len(set(names)) != len(names):
        raise ValueError(f'{split}: duplicate image basenames would overwrite output masks.')
    savedir = Path(args.savedir) / split
    savedir.mkdir(parents=True, exist_ok=True)
    metric = SalEval(device=device)
    mean, std = [0.406, 0.456, 0.485], [0.225, 0.224, 0.229]
    for image_path, label_path in tqdm(rows, desc=split):
        image_path = resolve_manifest_path(root, image_path)
        label_path = resolve_manifest_path(root, label_path)
        depth_path = Path(str(image_path.with_suffix('')) + args.depth_postfix)
        image = cv2.imread(str(image_path))
        label = cv2.imread(str(label_path), 0)
        depth = cv2.imread(str(depth_path), 0)
        for name, value in [(image_path, image), (label_path, label), (depth_path, depth)]:
            if value is None:
                raise FileNotFoundError(f'Could not read {name}')
        if image.shape[:2] != label.shape or depth.shape != label.shape:
            raise ValueError(f'Input dimensions differ for {image_path}')
        # Preserve the uploaded test preprocessing, including operation order.
        depth = (depth / 255.0 - 0.5) / 0.5
        depth = cv2.resize(depth, (args.inWidth, args.inHeight))
        depth = torch.from_numpy(depth).float()[None, None].to(device)
        img = cv2.resize(image, (args.inWidth, args.inHeight)).astype(np.float32) / 255.0
        img -= mean
        img /= std
        img = img[:, :, ::-1].copy().transpose(2, 0, 1)
        rgb = torch.from_numpy(img)[None].to(device)
        target = torch.from_numpy(label / 255.0).float()[None, None].to(device)
        prediction = model(rgb, depth)[:, :1]
        prediction = F.interpolate(prediction, size=image.shape[:2], mode='bilinear', align_corners=False)
        metric.addBatch(prediction[:, 0], target)
        mask = (prediction[0, 0].cpu().numpy() * 255).astype(np.uint8)
        if not cv2.imwrite(str(savedir / (image_path.stem + '.png')), mask):
            raise OSError(f'Could not save prediction for {image_path}')
    f_beta, mae = metric.getMetric()
    result = {'images': len(rows), 'max_F_beta_original_evaluator': f_beta, 'MAE_float_predictions': mae}
    print(split, json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--pretrained', required=True, help='Original or converted LiSalNet checkpoint.')
    parser.add_argument('--savedir', default='./maps/LiSalNet_updated')
    parser.add_argument('--inWidth', type=int, default=320)
    parser.add_argument('--inHeight', type=int, default=320)
    parser.add_argument('--gpu', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--depth_postfix', default='_depth.png')
    parser.add_argument('--lists', nargs='+', default=['NJU2K_test', 'NLPR_test', 'SSD', 'STERE', 'SIP', 'DES'])
    args = parser.parse_args()
    device = torch.device('cuda:0' if args.gpu else 'cpu')
    if args.gpu and not torch.cuda.is_available():
        parser.error('CUDA unavailable; pass --gpu False for CPU evaluation.')
    model = LiSalNet(pretrained=False)
    load_checkpoint(model, args.pretrained)
    model = model.to(device).eval()
    results = {split: evaluate(model, args, split, device) for split in args.lists}
    output = Path(args.savedir) / 'evaluation.json'
    output.write_text(json.dumps({'checkpoint': str(args.pretrained), 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
