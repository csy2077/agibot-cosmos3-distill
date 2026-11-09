#!/usr/bin/env bash
# Wait for the paired exports, then generate with each (FSDP over two GPUs with >= 30 GB free, one model
# after the other) and compare against GT.
set -uo pipefail
while kill -0 $(cat /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/export_pair.pid) 2>/dev/null; do sleep 30; done
for d in stage1-iter300 stage1-init; do
  [[ -f /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/checkpoints/clip15-test2-$d-export/model.safetensors.index.json ]] || { echo "export $d FAILED"; exit 1; }
done
pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2>=30720{print $2" "$1}' | sort -rn | awk '{print $2}' | head -n 2 | paste -sd,; }
for run in iter300:pred_s1_iter300_ar init:pred_s1_init_ar; do
  name=${run%%:*}; out=${run#*:}
  while [[ $(pick | tr , '\n' | wc -l) -lt 2 ]]; do sleep 60; done
  G=$(pick); echo "$name on GPUs $G"
  GPU=$G CKPT=/nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/checkpoints/clip15-test2-stage1-$name-export OUT=/nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/outputs/clip15_test2_gen/$out \
    /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/scripts/run_clip15_gen_ar.sh > /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/gen_s1_$name.log 2>&1 || { echo "gen $name FAILED"; tail -n 20 /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/train_runs/gen_s1_$name.log; exit 1; }
done
cd /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni && source /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos/cosmos-framework/.venv/bin/activate
python scripts/compare_clip_gen.py /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/outputs/clip15_test2_gen /nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos3_vllm_omni/outputs/clip15_test2_gen/compare_stage1 init=pred_s1_init_ar trained=pred_s1_iter300_ar
