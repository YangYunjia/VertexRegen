#!/bin/bash
# Fine-tune OPT-125M on bracketv3 seeded vsplit (F14 step 1).
# Usage: GPU=0 ./scripts/launch_arm8_125M_ft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
PY="${PY:-/home/yunjia/miniconda3/envs/armesh/bin/python}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libgomp.so.1}"
export PYTHONPATH="/home/yunjia/.deps_vertexregen:/mnt/netdisk1/yunjia/optShapeGen:${PYTHONPATH:-.}"

# Fail fast if weights are missing (this host's HF cache had config.json only).
if ! $PY - <<'PY'
from transformers import OPTForCausalLM
OPTForCausalLM.from_pretrained("facebook/opt-125m", local_files_only=True)
print("[ok] facebook/opt-125m weights present locally")
PY
then
  echo "ERROR: facebook/opt-125m weights not on disk."
  echo "  huggingface-cli download facebook/opt-125m"
  echo "or temporarily drop this check and let from_pretrained fetch."
  exit 1
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG="/tmp/r1_arm8_125M_ft_${RUN_TS}.log"
echo "launching on cuda:$GPU  RUN_TS=$RUN_TS  log=$LOG"
CUDA_VISIBLE_DEVICES="$GPU" RUN_TS="$RUN_TS" \
  nohup $PY -u train.py --config-name round1_arm8_125M_ft \
  > "$LOG" 2>&1 &
echo "PID=$!"
echo "tail -f $LOG"
