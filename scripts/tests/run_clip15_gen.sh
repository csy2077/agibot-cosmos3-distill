#!/usr/bin/env bash
# Forward-dynamics generation from the Stage 0 test export on the 2 training clips (1 GPU).
# Stage 0 is bidirectional (not distilled): full-sequence sampling, no CFG, training shift (480p -> 5).
set -euo pipefail
ROOT=/nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni
cd /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos/cosmos-framework
source .venv/bin/activate
# COSMOS_TRAINING=1 exposes --guidance/--shift on the CLI (they are hidden in inference-only mode).
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 HF_HOME=$ROOT/hf
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CKPT=${CKPT:-$ROOT/checkpoints/clip15-test2-stage0-iter300-export}
OUT=${OUT:-$ROOT/outputs/clip15_test2_gen/pred_iter300}
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference \
  -i ${MANIFEST:-$ROOT/outputs/clip15_test2_gen/inputs/manifest.jsonl} -o "$OUT" \
  --checkpoint-path "$CKPT" \
  --model-mode=FORWARD_DYNAMICS --resolution=480 --image-size=480 \
  --num-steps=${STEPS:-35} --guidance=1.0 --shift=5.0 --seed=2026 --autoregressive=False \
  --parallelism-preset=latency --max-num-seqs=1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-diffusion-cache --no-guardrails --video-save-quality=8 "$@"
