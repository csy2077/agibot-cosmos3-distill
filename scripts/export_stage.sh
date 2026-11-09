#!/usr/bin/env bash
# Export a training checkpoint to an HF-format inference checkpoint.
# Usage: CKPT=<run>/checkpoints/iter_XXXXXXXXX RUN_DIR=<run> STAGE=0|1|2|3 OUT=<export dir> scripts/export_stage.sh
# The run's own config.yaml is used, so the export carries that stage's attention/sampling setup.
# EMA=1 exports the EMA weights (only if the run trained with EMA on).
set -euo pipefail
: "${COSMOS_FRAMEWORK_DIR:?}" "${CKPT:?}" "${RUN_DIR:?}" "${STAGE:?}" "${OUT:?}"
cd "$COSMOS_FRAMEWORK_DIR"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
EXP=$(grep '^experiment' examples/toml/sft_config/$(ls examples/toml/sft_config | grep "^sim_bimanual_stage${STAGE}_") | cut -d'"' -f2)
EMA_FLAG=--no-use-ema-weights; [[ "${EMA:-0}" == "1" ]] && EMA_FLAG=--use-ema-weights
python -m cosmos_framework.scripts.export_model --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" \
  --experiment "$EXP" $EMA_FLAG --no-vit -o "$OUT"
