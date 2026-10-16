#!/bin/bash
# Train the FairSeg baselines once each (TransUNet, then SAMed) on 2 x RTX 3060 12GB.
# The sensitive attribute only groups the test metrics; baseline training does not use
# it, so one run per model serves all attributes (train.sh loops over them needlessly).
#
# Usage (from the repo root):  bash scripts/train_baselines.sh <dataset dir>
# Logs go to outputs/baseline/.
#
# Settings follow the official code (train.sh + train.py defaults), except where noted.
set -u
DATA=${1:?usage: train_baselines.sh <folder containing the npz files>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/Scripts/python.exe"
LOGS="$REPO/outputs/baseline"
mkdir -p "$LOGS"
echo "commit $(git -C "$REPO" rev-parse HEAD)  started $(date)" | tee -a "$LOGS/runs.txt"

# TransUNet: official train.sh uses batch 24 on 1 GPU with lr 0.01, SGD, 300 epochs.
# Here 2 GPUs x 12 = 24. train.py rescales lr by (per-GPU batch)/24, so pass 0.02 to
# end up at 0.01 (0.02 * 12/24).
cd "$REPO/TransUNet"
"$PY" train.py --root_path "$DATA" --list_dir ./lists/FairSeg_final \
    --num_classes 3 --img_size 224 --center_crop_size 512 \
    --max_epochs 300 --batch_size 12 --n_gpu 2 --base_lr 0.02 \
    > "$LOGS/transunet_train.log" 2>&1
echo "transunet exit $?  finished $(date)" | tee -a "$LOGS/runs.txt"

# SAMed: official train.sh uses 2 GPUs x 24 = 48; 12 GB cards fit 8 per GPU, so 16.
# Everything else as in train.sh: lr 0.005, warmup, AdamW, max 100 epochs, stop at 70.
cd "$REPO/SAMed"
"$PY" train.py --root_path "$DATA" --list_dir ./lists/FairSeg_final \
    --output ../outputs/baseline/samed \
    --num_classes 2 --img_size 512 --center_crop_size 512 \
    --max_epochs 100 --stop_epoch 70 --warmup --AdamW \
    --batch_size 8 --n_gpu 2 \
    > "$LOGS/samed_train.log" 2>&1
echo "samed exit $?  finished $(date)" | tee -a "$LOGS/runs.txt"
