#!/bin/bash
# TransUNet baseline only, with the same settings as scripts/train_baselines.sh.
# Usage (from the repo root):  bash scripts/train_transunet.sh <dataset dir> [log tag]
# The tag (default "transunet") names the log file and the line written to runs.txt.
set -u
DATA=${1:?usage: train_transunet.sh <folder containing the npz files> [log tag]}
TAG=${2:-transunet}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/Scripts/python.exe"
LOGS="$REPO/outputs/baseline"
mkdir -p "$LOGS"
dirty=$(git -C "$REPO" status --porcelain --untracked-files=no | grep -q . && echo " (uncommitted changes)")
echo "$TAG commit $(git -C "$REPO" rev-parse HEAD)$dirty  started $(date)" | tee -a "$LOGS/runs.txt"

# Official train.sh: batch 24, lr 0.01, SGD, 300 epochs. Here 2 GPUs x 12 = 24; train.py
# rescales lr by (per-GPU batch)/24, so 0.02 * 12/24 = 0.01.
cd "$REPO/TransUNet"
"$PY" train.py --root_path "$DATA" --list_dir ./lists/FairSeg_final \
    --num_classes 3 --img_size 224 --center_crop_size 512 \
    --max_epochs 300 --batch_size 12 --n_gpu 2 --base_lr 0.02 \
    > "$LOGS/${TAG}_train.log" 2>&1
echo "$TAG exit $?  finished $(date)" | tee -a "$LOGS/runs.txt"
