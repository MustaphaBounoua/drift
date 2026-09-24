#!/usr/bin/env bash
# Train and evaluate DRIFT on one dataset, one seed after another.
#
#   ./reproduce.sh <additive|holdout|replogle|combosciplex> [seeds] [gpu]
#   ./reproduce.sh additive 0,1,2,3,4 0
#
# Checkpoints go to checkpoints/, metrics to results/<dataset>-s<seed>.json.
set -euo pipefail
cd "$(dirname "$0")"
DATASET="$1"; SEEDS="${2:-0,1,2,3,4}"; export CUDA_VISIBLE_DEVICES="${3:-0}"
PY="${PYTHON:-python}"

for SEED in ${SEEDS//,/ }; do
  RUN="${DATASET}-s${SEED}"
  "$PY" scripts/train_vae.py --run-name "${RUN}-vae" --dataset "$DATASET" --seed "$SEED"
  "$PY" scripts/train_flow.py --run-name "$RUN" --vae-ckpt "checkpoints/${RUN}-vae/last.ckpt" \
    --seed "$SEED"
  "$PY" scripts/evaluate.py "checkpoints/${RUN}/last.ckpt" --json "results/${RUN}.json"
done
