#!/usr/bin/env bash
# Causal (autoregressive) forward-dynamics generation for Stage 1 test exports on the 2 training clips (GPU list in GPU; several GPUs are FSDP-sharded).
# Stage 1 is causal teacher forcing but not distilled: many sampling steps, no CFG, training shift (480p -> 5).
set -euo pipefail
ROOT=/nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni
cd /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos/cosmos-framework
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 HF_HOME=$ROOT/hf
export CUDA_VISIBLE_DEVICES=${GPU:-0}
NP=$(echo "$CUDA_VISIBLE_DEVICES" | tr , "\n" | wc -l)  # >1 GPU: FSDP-shard the weights across them
: "${CKPT:?set CKPT}" "${OUT:?set OUT}"
torchrun --standalone --nproc-per-node=$NP --master-port=${PORT:-29511} -m cosmos_framework.scripts.inference \
  -i ${MANIFEST:-$ROOT/outputs/clip15_test2_gen/inputs/manifest.jsonl} -o "$OUT" \
  --checkpoint-path "$CKPT" \
  --model-mode=FORWARD_DYNAMICS --resolution=480 --image-size=480 \
  --num-steps=${STEPS:-35} --guidance=1.0 --shift=5.0 --seed=2026 \
  --autoregressive=True --kv-cache-inference-size=64 --attention-sink-size=8 \
  --parallelism-preset=latency --dp-shard-size=$NP --cp-size=1 --max-num-seqs=1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-diffusion-cache --no-guardrails --video-save-quality=8 "$@"
