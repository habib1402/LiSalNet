# Checkpoint release map

Checkpoint binaries are distributed through the
[public release-assets folder](https://drive.google.com/drive/folders/1x7qqC_t9E4M1e5Dpu5na4FOVPDFjNJcs?usp=sharing)
and are intentionally ignored by Git.

| Release filename | Role | Paper datasets |
|---|---|---|
| `LiSalNet_single_checkpoint.pth` | Recommended one-checkpoint baseline | all six |
| `LiSalNet_paper_run_v1.pth` | Paper-selected run v1 | DES, SIP |
| `LiSalNet_paper_run_v2.pth` | Paper-selected run v2 | NJU2K, STERE |
| `LiSalNet_paper_run_v3.pth` | Paper-selected run v3 | NLPR |
| `LiSalNet_paper_run_v4.pth` | Paper-selected run v4 | SSD |

Every file has the `LiSalNet_without_RGB_block_18` architecture marker and
3,984,607 parameters. The conversion keeps every active tensor strictly and
removes only the unused MobileNetV2 terminal projection entries. Release
archives include a SHA-256 manifest.

| File | SHA-256 |
|---|---|
| `LiSalNet_single_checkpoint.pth` | `b55a105c42b549ee35fb0a781d5ad1d7dd8f97e862b3396cba9b1a72db26e1d5` |
| `LiSalNet_paper_run_v1.pth` | `913b479030c09731d0a0f0b07ffb3acd97bed8fae7251c50ab607e691cf861f2` |
| `LiSalNet_paper_run_v2.pth` | `2e4f5a28fc4fee5c83f08638a77d2c0ac0e429ecc9be35ce5c7362b117869ac7` |
| `LiSalNet_paper_run_v3.pth` | `5093677a517a8df57b818479715ad43813a3c6461292abcc14594a7ac3a8240f` |
| `LiSalNet_paper_run_v4.pth` | `2e763b86dc8e0d19b4835dbe27ef5d5bc74e7c362c5b9dc283d1e39b3ce00f0b` |

The paper explicitly reports benchmark-wise best results across six independent
runs; therefore, use the mapping above only when reproducing the published
table. Use `LiSalNet_single_checkpoint.pth` for a single-model comparison.
