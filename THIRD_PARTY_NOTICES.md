# Third-party notices

LiSalNet depends on or interoperates with the following third-party projects.
They are not relicensed by LiSalNet's MIT license.

## PyTorch and torchvision

The MobileNetV2 encoder structure follows the torchvision implementation and
loads the official PyTorch ImageNet weights. PyTorch and torchvision are
distributed under the BSD 3-Clause license. Copyright remains with their
respective contributors.

- https://github.com/pytorch/pytorch
- https://github.com/pytorch/vision

## PySODMetrics

`tools/evaluate_maps.py` uses PySODMetrics, Copyright (c) 2020 lartpang,
distributed under the MIT License. It is installed as a dependency and is not
vendored in this repository.

- https://github.com/lartpang/PySODMetrics

## Depth Anything V2

`tools/generate_pseudo_depth.py` interoperates with a separate checkout of
Depth Anything V2. No Depth Anything source or weights are redistributed here.
The upstream project states that its Small model is Apache-2.0, while its
Base/Large/Giant model weights are CC-BY-NC-4.0. LiSalNet's published
pseudo-depth preprocessing uses the Large (`vitl`) weights, so users must comply
with that non-commercial model license.

- https://github.com/DepthAnything/Depth-Anything-V2
