#!/usr/bin/env bash
# Short Stage 0 overfit test on the 2-clip 15 fps bundle: GPUs 0,1, FSDP 2 x CP 1, 1 sample per GPU (global 2).
# Usage: run_stage0_clip15_test.sh [--dryrun] [extra hydra overrides...]
set -euo pipefail
ROOT=/nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni
cd /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos/cosmos-framework
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
export HF_HOME=$ROOT/hf
export COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE=$ROOT/outputs/clip15_test2/train
export COSMOS_SIM_BIMANUAL_GA_CHECKPOINT=$ROOT/checkpoints/Cosmos3-Nano-dcp-e59a53c
export IMAGINAIRE_OUTPUT_ROOT=$ROOT/train_runs
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
TOML=examples/toml/sft_config/sim_bimanual_stage0_bidir.toml
DRY=()
if [[ "${1:-}" == "--dryrun" ]]; then DRY=(--dryrun); shift; fi
OVR=(
  "job.name=${RUN_NAME:-clip15-test2-stage0}"
  "model.config.tokenizer.vae_path=$ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth"
  model.config.parallelism.data_parallel_shard_degree=2
  model.config.parallelism.context_parallel_shard_degree=1
  model_parallel.context_parallel_size=1
  dataloader_train.max_samples_per_batch=1
  trainer.max_iter=${MAX_ITER:-300}
  trainer.logging_iter=5
  checkpoint.save_iter=${SAVE_ITER:-300}
  "$@"
)
if [[ ${#DRY[@]} -gt 0 ]]; then
  python -m cosmos_framework.scripts.train --sft-toml="$TOML" --dryrun -- "${OVR[@]}"
else
  torchrun --standalone --nproc-per-node=2 -m cosmos_framework.scripts.train --sft-toml="$TOML" -- "${OVR[@]}"
fi
