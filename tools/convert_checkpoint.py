#!/usr/bin/env python3
"""Export original LiSalNet weights for the model without RGB block 18.

The destination contains model weights only. Legacy Adam state has parameter
indices for the old architecture and is deliberately not copied.
"""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from checkpoint_utils import load_checkpoint
from models import LiSalNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("Destination already exists; choose a new output path.")
    model = LiSalNet(pretrained=False).cpu().eval()
    removed = load_checkpoint(model, args.source)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as handle:
        torch.save({
            "state_dict": model.state_dict(),
            "architecture": "LiSalNet_without_RGB_block_18",
            "parameters": sum(p.numel() for p in model.parameters()),
            "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
            "removed_keys": removed,
            "weights_only_export": True,
        }, handle)
    print(f"Saved {args.destination}; 3,984,607 model parameters; no optimizer state.")


if __name__ == "__main__":
    main()
