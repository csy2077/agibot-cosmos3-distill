#!/usr/bin/env bash
# Forward-dynamics generation (first frame + actions -> video) from an exported checkpoint.
# Usage: CKPT=<export> MANIFEST=<rows.jsonl> OUT=<dir> GPU=0[,1] MODE=ar|full STEPS=35 scripts/generate.sh
#   MODE=full : whole-clip sampling (Stage 0, bidirectional)
#   MODE=ar   : causal chunk-by-chunk sampling (Stages 1-3); a distilled Stage 3 student uses STEPS=4
# Several GPUs in GPU shard the weights with FSDP (use when one GPU lacks memory).
set -euo pipefail
: "${COSMOS_FRAMEWORK_DIR:?}" "${CKPT:?}" "${MANIFEST:?}" "${OUT:?}"
cd "$COSMOS_FRAMEWORK_DIR"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1  # TRAINING=1 exposes --guidance/--shift
export CUDA_VISIBLE_DEVICES=${GPU:-0}
NP=$(echo "$CUDA_VISIBLE_DEVICES" | tr , '\n' | wc -l)
if [[ "${MODE:-ar}" == "ar" ]]; then
  AR=(--autoregressive=True --kv-cache-inference-size=64 --attention-sink-size=8)
else
  AR=(--autoregressive=False)
fi
torchrun --standalone --nproc-per-node=$NP --master-port=${PORT:-29511} -m cosmos_framework.scripts.inference \
  -i "$MANIFEST" -o "$OUT" --checkpoint-path "$CKPT" \
  --model-mode=FORWARD_DYNAMICS --resolution=480 --image-size=480 \
  --num-steps=${STEPS:-35} --guidance=${GUIDANCE:-1.0} --shift=5.0 --seed=2026 "${AR[@]}" \
  --parallelism-preset=latency --dp-shard-size=$NP --cp-size=1 --max-num-seqs=1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-diffusion-cache --no-guardrails --video-save-quality=8 "$@"
