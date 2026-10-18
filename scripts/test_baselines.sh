#!/bin/bash
# Test the final SAMed and TransUNet baseline checkpoints on the official test list
# (lists/FairSeg_final/test.txt, 2000 images) for each sensitive attribute.
# SAMed runs on GPU 0 and TransUNet on GPU 1 at the same time.
#
# Usage (from the repo root):  bash scripts/test_baselines.sh <dataset dir>
# Results: outputs/baseline_test/{samed,transunet}/<checkpoint>/<attribute>.txt
set -u
DATA=${1:?usage: test_baselines.sh <folder containing the npz files>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/Scripts/python.exe"
OUT="$REPO/outputs/baseline_test"
ATTRS="race gender language ethnicity"
SAMED_CKPT="$REPO/outputs/baseline/samed/FairSeg_512_pretrain_vit_b_epo100_bs8_lr0.005/epoch_69.pth"
TU_CKPT="$REPO/model/TU_FairSeg224/TU_pretrain_ViT-B_16_skip0_epo300_bs12_224/epoch_299.pth"
mkdir -p "$OUT"
echo "commit $(git -C "$REPO" rev-parse HEAD)  started $(date)" >> "$OUT/runs.txt"

(
  cd "$REPO/SAMed"
  for a in $ATTRS; do
    CUDA_VISIBLE_DEVICES=0 "$PY" test.py --datadir "$DATA" --list_dir ./lists/FairSeg_final \
        --num_classes 2 --img_size 512 --center_crop_size 512 \
        --lora_ckpt "$SAMED_CKPT" --attribute "$a" \
        --output_dir "$OUT/samed/epoch_69_$a" > "$OUT/samed_$a.log" 2>&1
    echo "samed $a exit $?  $(date)" >> "$OUT/runs.txt"
  done
) &
(
  cd "$REPO/TransUNet"
  for a in $ATTRS; do
    CUDA_VISIBLE_DEVICES=1 "$PY" test.py --datadir "$DATA" --list_dir ./lists/FairSeg_final \
        --num_classes 3 --img_size 224 --center_crop_size 512 \
        --snapshot "$TU_CKPT" --attribute "$a" \
        --output_dir "$OUT/transunet" > "$OUT/transunet_$a.log" 2>&1
    echo "transunet $a exit $?  $(date)" >> "$OUT/runs.txt"
  done
) &
wait
echo "all tests finished $(date)" >> "$OUT/runs.txt"
