#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
if [ "$#" -lt 2 ]; then
    echo 'Usage: bash test.sh CHECKPOINT DATA_ROOT [additional test.py arguments]'
    exit 2
fi
TASK_CHECKPOINT="$1"
TASK_DATA_ROOT="$2"
shift 2
python3 tools/test.py --pretrained "$TASK_CHECKPOINT" --data_dir "$TASK_DATA_ROOT" "$@"
