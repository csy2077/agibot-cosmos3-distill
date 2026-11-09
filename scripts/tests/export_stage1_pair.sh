#!/usr/bin/env bash
# Export the Stage 1 test checkpoint (iter 300) and its initialization (Stage 0 iter 300 weights) under the
# same Stage 1 config, so both run through the identical causal model and inference path.
set -uo pipefail
cd /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos/cosmos-framework
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 HF_HOME=/nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/hf
exp() { python -m cosmos_framework.scripts.export_model --checkpoint-path "$1" --config-file /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/cosmos3_simulation/bimanual/clip15-test2-stage1/config.yaml \
  --experiment sim_local_bimanual_stage1_tf --no-use-ema-weights --no-vit -o "$2"; }
exp /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/cosmos3_simulation/bimanual/clip15-test2-stage1/checkpoints/iter_000000300 /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/checkpoints/clip15-test2-stage1-iter300-export > /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/export_stage1_iter300.log 2>&1 &
exp /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/cosmos3_simulation/bimanual/clip15-test2-stage0/checkpoints/iter_000000300 /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/checkpoints/clip15-test2-stage1-init-export > /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/export_stage1_init.log 2>&1 &
wait
