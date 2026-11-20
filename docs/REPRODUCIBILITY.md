# Reproducibility notes

This document records the implementation and reporting choices needed to
interpret the public LiSalNet artifacts against the final published paper.

## Architecture identity

The released network has 3,984,607 registered parameters. Its MobileNetV2 RGB
feature encoder ends at block 17 (320 channels); the classifier-oriented block
18 (320→1280) is not registered or executed. The released architecture and
legacy-checkpoint conversion were numerically verified at 256, 288, and 320
pixels, including forward equivalence after removal of the unused block,
back-propagation through all trainable parameters, and shared-DPM execution.

| Module | Parameters |
|---|---:|
| MobileNetV2 RGB encoder | 1,811,712 |
| Depth Feature Extractor | 1,130,620 |
| MAN1 + MAN5 | 725,752 |
| PFU1 + PFU3 + PFU5 | 172,032 |
| SFD | 138,816 |
| DPM3 | 2,808 |
| SFB3 | 2,496 |
| Prediction heads | 371 |
| **Total** | **3,984,607** |

## Data and preprocessing

- Training: 1,485 NJU2K-train + 700 NLPR-train samples.
- Testing: NJU2K (500), NLPR (300), SSD (80), STERE (1,000), SIP
  (929), and DES (135).
- Main results: pseudo-depth for both training and testing, generated using
  Depth Anything V2-Large (`vitl`) at input size 518, followed by per-image
  min–max conversion to an 8-bit grayscale PNG.
- RGB normalization is applied to OpenCV BGR values before channel reversal,
  using means `(0.406, 0.456, 0.485)` and standard deviations
  `(0.225, 0.224, 0.229)`. Depth is mapped from `[0, 255]` to `[-1, 1]`.
- Training augmentation uses a random symmetric margin crop/resize and a
  vertical flip, independently with probability 0.5. Masks use nearest-neighbor
  interpolation; RGB and depth use bilinear interpolation.

## Training and checkpoint selection

The final comparison model is trained for 150 epochs. Each epoch contains three
complete passes at 256, 288, and 320 pixels, producing 98,100 optimizer steps
for 2,185 samples at batch size 10 with incomplete batches dropped.

The paper states: “Results are reported from the best of six independent
training runs for each benchmark.” Consequently, the paper table does not
represent one universal checkpoint. The released paper-result assets preserve
that protocol transparently:

| Dataset | Paper-result run |
|---|---|
| DES, SIP | v1 |
| NJU2K, STERE | v2 |
| NLPR | v3 |
| SSD | v4 |

All four released run checkpoints strictly load into the 3.98M architecture.
An unrelated historical NJU2K file containing an additional `idr` module is not
part of the release. `LiSalNet_single_checkpoint.pth` is the recommended default
for method comparison because it is one epoch-150 checkpoint evaluated on all
six datasets without benchmark-wise selection.

### Single-checkpoint baseline

These values were calculated from the saved uint8 prediction maps with the
project evaluator:

| Dataset | MAE ↓ | S ↑ | max F ↑ | max E ↑ |
|---|---:|---:|---:|---:|
| NJU2K | 0.0341 | 0.9192 | 0.9244 | 0.9524 |
| NLPR | 0.0221 | 0.9270 | 0.9165 | 0.9624 |
| STERE | 0.0348 | 0.9153 | 0.9094 | 0.9479 |
| SIP | 0.0449 | 0.8917 | 0.9025 | 0.9306 |
| SSD | 0.0504 | 0.8607 | 0.8443 | 0.9096 |
| DES | 0.0172 | 0.9385 | 0.9424 | 0.9769 |

The inference script's legacy MAE/max-F summary operates on float predictions.
The complete evaluator operates on saved uint8 PNGs. Use the latter when
comparing with the supplied prediction archives. The public evaluator pins
PySODMetrics 1.6.2; tiny last-decimal differences from the historical project
evaluator may occur, so the supplied PNGs and reported paper values remain the
authoritative release artifacts.

## Efficiency protocol

The published LiSalNet row was measured with Python 3.8.10, PyTorch
2.1.1+cu121, CUDA 12.1, cuDNN 8902, and THOP 0.1.1-2209072238 on an RTX 4090.
The model and random inputs were resident on the GPU. FP32 and batch size 1 were
used; TF32 and cuDNN benchmarking were disabled; `cudnn.deterministic=True` and
`torch.use_deterministic_algorithms(False)`. Timing used 60 warm-up and 300
synchronized calls in each of three fresh processes.

- Latency (8.84 ms): median synchronized latency of the median run.
- FPS (113.2): reciprocal of that median latency, not batched throughput.
- GMACs (1.284614848): THOP reported count at 320×320.
- GFLOPs (2.569229696): twice the THOP count, as stated in the paper.
- Model size (15.3941459656 MiB): parameter plus registered-buffer payload in
  FP32. This is a payload definition, not the on-disk checkpoint file size.
- Peak VRAM (21.51416015625 MiB): additional PyTorch-allocated memory above the
  resident model and inputs during one forward. Total peak allocated memory was
  38.658203125 MiB.
- Included: the complete three-head model forward.
- Excluded: image/depth loading, resizing, pseudo-depth generation, host-device
  transfers, and map saving.

Latency is software- and hardware-sensitive. Use the operation and parameter
counts for portable structural checks, and do not mix timing values measured
under different PyTorch/CUDA versions in one comparison table.
