#!/usr/bin/env python3
"""Generate the grayscale Depth Anything V2 pseudo-depth used by LiSalNet."""
from pathlib import Path
import argparse
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import resolve_manifest_path


CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, type=Path)
    parser.add_argument('--lists', nargs='+', required=True,
                        help='Manifest names without the .txt suffix.')
    parser.add_argument('--depth_anything_root', required=True, type=Path,
                        help='Official Depth-Anything-V2 repository checkout.')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--encoder', choices=CONFIGS, default='vitl')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--depth_postfix', default='_depth.png')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.input_size <= 0:
        parser.error('--input_size must be positive.')

    sys.path.insert(0, str(args.depth_anything_root))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as error:
        parser.error(f'Could not import Depth Anything V2 from {args.depth_anything_root}: {error}')

    device = torch.device(args.device)
    model = DepthAnythingV2(**CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model = model.to(device).eval()

    images = []
    for split in args.lists:
        manifest = args.data_dir / f'{split}.txt'
        rows = [line.split() for line in manifest.read_text().splitlines() if line.strip()]
        if not rows or any(len(row) != 2 for row in rows):
            raise ValueError(f'{manifest}: expected nonempty image_path mask_path pairs.')
        images.extend(resolve_manifest_path(args.data_dir, row[0]) for row in rows)
    images = list(dict.fromkeys(images))

    for image_path in tqdm(images, desc='pseudo-depth'):
        output = Path(str(image_path.with_suffix('')) + args.depth_postfix)
        if output.exists() and not args.overwrite:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Could not read RGB image: {image_path}')
        depth = model.infer_image(image, args.input_size)
        low, high = float(depth.min()), float(depth.max())
        if high > low:
            depth = (depth - low) / (high - low) * 255.0
        else:
            depth = np.zeros_like(depth)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), depth.astype(np.uint8)):
            raise OSError(f'Could not write pseudo-depth: {output}')


if __name__ == '__main__':
    main()
