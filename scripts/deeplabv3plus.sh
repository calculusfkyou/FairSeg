#!/bin/bash
# Run a DeepLabV3+ stage (DeepLabv3+/train.py, bda.py or test.py) with --root_path filled in.
# Usage (from the repo root):  bash scripts/deeplabv3plus.sh <dataset dir> <train|bda|test> <tag> [more args]
# Output is appended (not overwritten) to outputs/deeplabv3plus/<tag>.log; the start (with the
# full command) and the end are written to outputs/deeplabv3plus/runs.txt, as in train_transunet.sh.
set -u
DATA=${1:?usage: deeplabv3plus.sh <dataset dir> <train|bda|test> <tag> [more args]}
STAGE=${2:?usage: deeplabv3plus.sh <dataset dir> <train|bda|test> <tag> [more args]}
TAG=${3:?usage: deeplabv3plus.sh <dataset dir> <train|bda|test> <tag> [more args]}
shift 3
case "$STAGE" in train|bda|test) ;; *) echo "stage must be train, bda or test, not '$STAGE'" >&2; exit 2 ;; esac
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/Scripts/python.exe"
LOGS="$REPO/outputs/deeplabv3plus"
mkdir -p "$LOGS"
dirty=$(git -C "$REPO" status --porcelain --untracked-files=no | grep -q . && echo " (uncommitted changes)")
echo "$TAG commit $(git -C "$REPO" rev-parse HEAD)$dirty  started $(date)  cmd: cd DeepLabv3+ && python $STAGE.py --root_path $(printf '%q' "$DATA")$(printf ' %q' "$@")" \
    | tee -a "$LOGS/runs.txt" "$LOGS/$TAG.log"

cd "$REPO/DeepLabv3+"
"$PY" "$STAGE.py" --root_path "$DATA" "$@" >> "$LOGS/$TAG.log" 2>&1
echo "$TAG exit $?  finished $(date)" | tee -a "$LOGS/runs.txt" "$LOGS/$TAG.log"
