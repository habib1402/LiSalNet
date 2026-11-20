#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
if [ "$#" -lt 1 ]; then
    echo 'Usage: bash train.sh DATA_ROOT [additional train.py arguments]'
    exit 2
fi
TASK_DATA_ROOT="$1"
shift
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 tools/train.py \
    --file_root "$TASK_DATA_ROOT" --max_epochs 150 --batch_size 10 \
    --lr 1e-4 --inWidth 320 --inHeight 320 "$@"
