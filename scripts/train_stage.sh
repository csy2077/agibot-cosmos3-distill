#!/usr/bin/env bash
# Train one Sim-Bimanual distillation stage (0-3) on one node.
# Default: 4 GPUs, FSDP 4 x CP 1, 1 sample per GPU, gradient accumulation 4 -> global batch 16.
#
# Usage: STAGE=0 scripts/train_stage.sh [--dryrun] [extra hydra overrides...]
# Required env: COSMOS_FRAMEWORK_DIR, BUNDLE (bundle train/ dir), NANO_DCP (Cosmos3-Nano DCP root),
#               VAE_PTH (Wan2.2_VAE.pth), OUTPUT_ROOT (checkpoint/log root)
# Stage handoffs (checkpoint ITERATION ROOTS, e.g. .../checkpoints/iter_000035000):
#   STAGE 1 needs STAGE0_CKPT; STAGE 2 needs STAGE1_CKPT; STAGE 3 needs STAGE0_CKPT and STAGE2_CKPT.
# Optional env: GPUS (0,1,2,3), GLOBAL_BATCH (16), MAX_ITER, SAVE_ITER (500), RUN_NAME, EMA (0|1, default 0)
set -euo pipefail
STAGE=${STAGE:?set STAGE=0|1|2|3}
: "${COSMOS_FRAMEWORK_DIR:?}" "${BUNDLE:?}" "${NANO_DCP:?}" "${VAE_PTH:?}" "${OUTPUT_ROOT:?}"
GPUS=${GPUS:-0,1,2,3}
NGPU=$(echo "$GPUS" | tr , '\n' | wc -l)
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
(( GLOBAL_BATCH % NGPU == 0 )) || { echo "GLOBAL_BATCH ($GLOBAL_BATCH) must be divisible by the GPU count ($NGPU)"; exit 1; }
ACCUM=$(( GLOBAL_BATCH / NGPU ))  # 1 sample per GPU per micro-step
case $STAGE in  # reference budgets from the Framework canonical recipe
  0) DEF_ITER=35000 ;; 1) DEF_ITER=8000 ;; 2) DEF_ITER=4000 ;; 3) DEF_ITER=1500 ;;
  *) echo "STAGE must be 0-3"; exit 1 ;;
esac

cd "$COSMOS_FRAMEWORK_DIR"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
export COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE=$BUNDLE
export COSMOS_SIM_BIMANUAL_GA_CHECKPOINT=$NANO_DCP
export IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT
export CUDA_VISIBLE_DEVICES=$GPUS
case $STAGE in
  1) export COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT=${STAGE0_CKPT:?STAGE 1 needs STAGE0_CKPT} ;;
  2) export COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT=${STAGE1_CKPT:?STAGE 2 needs STAGE1_CKPT} ;;
  3) export COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT=${STAGE0_CKPT:?STAGE 3 needs STAGE0_CKPT}
     export COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT=${STAGE2_CKPT:?STAGE 3 needs STAGE2_CKPT} ;;
esac
TOML=examples/toml/sft_config/$(ls examples/toml/sft_config | grep "^sim_bimanual_stage${STAGE}_")

OVR=(
  "job.name=${RUN_NAME:-agibot-clip15-stage$STAGE}"
  "model.config.tokenizer.vae_path=$VAE_PTH"
  "model.config.parallelism.data_parallel_shard_degree=$NGPU"
  model.config.parallelism.context_parallel_shard_degree=1
  model_parallel.context_parallel_size=1
  dataloader_train.max_samples_per_batch=1
  "trainer.grad_accum_iter=$ACCUM"
  "trainer.max_iter=${MAX_ITER:-$DEF_ITER}"
  "checkpoint.save_iter=${SAVE_ITER:-500}"
  trainer.logging_iter=10
)
if [[ "${EMA:-0}" == "0" ]]; then
  # EMA keeps an extra fp32 copy of the weights; disabling it is what made the 2-GPU tests fit.
  OVR+=(model.config.ema.enabled=false)
  [[ $STAGE == 0 ]] && OVR+=(trainer.callbacks.initialize_ema_from_regular.enabled=false)
fi
DRY=()
if [[ "${1:-}" == "--dryrun" ]]; then DRY=(--dryrun); shift; fi
OVR+=("$@")

echo "STAGE=$STAGE TOML=$TOML GPUS=$GPUS global_batch=$GLOBAL_BATCH (1 x $NGPU GPUs x accum $ACCUM) EMA=${EMA:-0}"
if [[ ${#DRY[@]} -gt 0 ]]; then
  python -m cosmos_framework.scripts.train --sft-toml="$TOML" --dryrun -- "${OVR[@]}"
else
  torchrun --standalone --nproc-per-node="$NGPU" -m cosmos_framework.scripts.train --sft-toml="$TOML" -- "${OVR[@]}"
fi
