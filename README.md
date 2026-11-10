# agibot-cosmos3-distill

Post-train **Cosmos3-Nano** into an action-conditioned bimanual world model (the *Cosmos3-Nano-Sim-Bimanual* recipe)
on **AgiBot World Beta** clips, using the four-stage recipe in
[NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework):

| Stage | What it does | Starts from | Framework TOML |
|---|---|---|---|
| 0 | Bidirectional action-conditioned video model (adapts Nano, trains fresh action modules) | Cosmos3-Nano | `sim_bimanual_stage0_bidir.toml` |
| 1 | Causal teacher forcing (chunks of 2 latent frames, three-way attention) | Stage 0 | `sim_bimanual_stage1_tf.toml` |
| 2 | Fixed-step distillation (TF-dCM) toward a 4-step sampler | Stage 1 (student and frozen teacher) | `sim_bimanual_stage2_tfdcm.toml` |
| 3 | Self-forcing rollout distillation (DMD); the final 4-step student | Stage 2 student; Stage 0 teacher/critic | `sim_bimanual_stage3_sfdmd.toml` |

This repository holds only what we added on top of the Framework: data conversion, launchers for a 4-GPU node,
a small Framework patch, and generation/evaluation helpers. The raw training data is on Hugging Face:
[`csusupergear/agibot-cosmos3-distill`](https://huggingface.co/datasets/csusupergear/agibot-cosmos3-distill).

## Status (what has actually been run)

| Step | Status |
|---|---|
| Data conversion and Framework data checks (static + per-stage materialization) | passed on the 2-clip test bundle; full build not yet run |
| Stage 0, 2-clip overfit test, 2 GPUs, 300 iters | loss 0.89 → 0.13 (median), checkpoint exported, generation checked |
| Stage 1, 2-clip overfit test, 2 GPUs, 300 iters (with the patch below) | loss 0.55 → 0.04 |
| Stage 2 and Stage 3 | **not run yet** |
| 4 GPUs, global batch 16, full data | **not run yet**: the launcher is ready and passed config dry runs, but memory at 4 GPUs is untested |

Tested on RTX PRO 6000 Blackwell (96 GB) GPUs.

## 1. Setup

### Framework

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
git checkout b2faa2617b21ad572f0ff2e8421d738ec0cb4a84      # the commit we used
git apply /path/to/agibot-cosmos3-distill/patches/0001-causal-attention-skip-short-segments.patch
uvx uv@0.11.29 sync --all-extras --group=cu130-train       # creates .venv
export COSMOS_FRAMEWORK_DIR=$PWD
```

**Why the patch:** in Stage 1+ teacher forcing, `three_way_attention_with_memory` pads empty K/V with a
1-token placeholder segment, then zeroes a 4-token null-action block at every segment start; on the 1-token
segment that indexes past the buffer (surfaces as a `NATTEN failure: device-side assert`). The patch skips
segments shorter than one action supertoken. Stage 0 does not use this code path.

If git complains about "dubious ownership" on a network filesystem, point `GIT_CONFIG_GLOBAL` at a file that
contains `[safe] directory = *` while running `uv sync`.

### Checkpoints

Follow the Framework's post-training guide
(`.agents/skills/cosmos3-nano-sim-bimanual/post-training/SKILL.md`) for the authoritative steps. In short:

```bash
# Cosmos3-Nano (Hugging Face) -> DCP training checkpoint. Run with network access (HF_HUB_OFFLINE breaks
# processor resolution).
hf download nvidia/Cosmos3-Nano --local-dir checkpoints/Cosmos3-Nano
python -m cosmos_framework.scripts.convert_model_to_dcp --checkpoint-path checkpoints/Cosmos3-Nano \
  -o checkpoints/Cosmos3-Nano-dcp
# Wan2.2 VAE: the ORIGINAL .pth file (the diffusers-format copy does not load)
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir checkpoints/wan22_vae
```

## 2. Data

The Hugging Face dataset [`csusupergear/agibot-cosmos3-distill`](https://huggingface.co/datasets/csusupergear/agibot-cosmos3-distill)
holds the raw sources, packed one tar per task: ~93k single-action clips (head camera, 640×480, 15 fps, with a
visual caption each), the full 30 fps robot state of their ~44k episodes, and AgiBot task metadata. You build
the Framework training bundle from them.

### Download and unpack

```bash
hf download csusupergear/agibot-cosmos3-distill --repo-type dataset --local-dir raw
mkdir -p data && for f in raw/clips/*.tar raw/proprio_stats/*.tar; do tar xf "$f" -C data; done
cp -r raw/task_info data/ && cp raw/clips_train_manifest.json data/clips/train_manifest.json
```

This gives `data/clips/<clip>/head/{rgb.mp4, caption.pickle, episode_meta.npz}`,
`data/proprio_stats/<task>/<episode>/proprio_stats.h5` and `data/task_info/task_<task>.json`.
`caption.pickle` needs `torch` to load (use the Framework's `.venv`).

### Build the training bundle

The builder also needs the published example bundle as a template (it supplies the normalization statistics
and tokenizer resources):

```bash
hf download nvidia/Cosmos3-Nano-Sim-Bimanual examples/training/sim_bimanual_robot_example_v1.tar.zst --local-dir hf_example
mkdir -p templates/tmp && tar --zstd -xf hf_example/examples/training/sim_bimanual_robot_example_v1.tar.zst -C templates/tmp
# --template must point at the extracted folder that contains manifest.json and resources/
find templates/tmp -maxdepth 3 -name manifest.json
```

```bash
export AGIBOT_CLIP_ROOT=$PWD/data/clips AGIBOT_PROCESSED_ROOT=$PWD/data
$COSMOS_FRAMEWORK_DIR/.venv/bin/python scripts/build_clip_bundle_full.py \
  --out bundle/clip15_full --template <folder containing manifest.json> --workers 16
BUNDLE=$PWD/bundle/clip15_full/train scripts/validate_bundle.sh
```

What the builder does, per clip:

- takes robot-state rows `2*(start+i)` for clip frame `i` (30 fps record → 15 fps clip; the clips are trusted
  to be cut on this grid, so `episode_meta.npz` is not needed);
- writes the state in the Framework's ACT2 action format (`agibotworld` domain; grippers stay in raw Robotiq
  units, which the Framework loader calibrates using its audited gripper manifest);
- trims the clip to 1 + 4k frames (minimum 33) and writes the caption as
  `<visual caption> [0.0s-Ts]: <action text>`;
- assigns episodes to `train` / `val` by hash (2% validation; all clips of an episode stay in one split).

The output `bundle/clip15_full/{train,val}` are Framework local training bundles (`manifest.json`,
`samples.json`, `assets/`, `resources/`, `receipts/`). Reruns resume where they stopped. Clips that fail (wrong video size or frame rate, too short, past the end
of the state record) are listed in `_work/skipped.json`, and episodes without an audited gripper label or state file in
`_work/unmapped.txt`. `summary.json` gives the final clip, episode and hour counts.

## 3. Train the four stages (4 GPUs, global batch 16)

```bash
export COSMOS_FRAMEWORK_DIR=/path/to/cosmos-framework
export BUNDLE=$PWD/bundle/clip15_full/train
export NANO_DCP=$PWD/checkpoints/Cosmos3-Nano-dcp
export VAE_PTH=$PWD/checkpoints/wan22_vae/Wan2.2_VAE.pth
export OUTPUT_ROOT=$PWD/runs
export GPUS=0,1,2,3 GLOBAL_BATCH=16
RUNS=$OUTPUT_ROOT/cosmos3_simulation/bimanual

STAGE=0 scripts/train_stage.sh
STAGE=1 STAGE0_CKPT=$RUNS/agibot-clip15-stage0/checkpoints/iter_000035000 scripts/train_stage.sh
STAGE=2 STAGE1_CKPT=$RUNS/agibot-clip15-stage1/checkpoints/iter_000008000 scripts/train_stage.sh
STAGE=3 STAGE0_CKPT=$RUNS/agibot-clip15-stage0/checkpoints/iter_000035000 \
        STAGE2_CKPT=$RUNS/agibot-clip15-stage2/checkpoints/iter_000004000 scripts/train_stage.sh
```

Pass checkpoint **iteration roots** (`.../checkpoints/iter_XXXXXXXXX`), not their `model/` subfolder. Add
`--dryrun` as the first argument to print the resolved config without training.

**Batch arithmetic.** Global batch = samples per GPU × data-parallel GPUs × gradient accumulation
(the Framework's own rule). The launcher fixes 1 sample per GPU, shards the model over the GPUs with FSDP
(no context parallelism) and sets `trainer.grad_accum_iter = GLOBAL_BATCH / #GPUs`, so 4 GPUs give 1 × 4 × 4 = 16.
Stage 0 normally packs samples by a token budget; the launcher pins it to one clip per GPU so the batch is
exactly 16 clips.

**Iterations.** Defaults are the Framework reference budgets: 35,000 / 8,000 / 4,000 / 1,500 for Stages 0–3.
Those were set for NVIDIA's own (larger) setup, so treat them as a starting point. Override with
`MAX_ITER=...`; checkpoints save every `SAVE_ITER` iterations (default 500). Each checkpoint is about 114 GB with
EMA off (more with EMA on).

**EMA.** The reference recipe keeps an EMA copy of the weights. The launcher defaults to `EMA=0` because the
2-GPU tests ran out of memory at the optimizer step with EMA on. With 4 GPUs each GPU holds half as much
optimizer state, so `EMA=1` may fit. Keep the choice the same across all stages, since Stage 3 reuses the Stage 0
checkpoint. Export with `EMA=1` only from a run trained with EMA.

**Memory.** Measured on the 2-GPU tests (1 sample/GPU, EMA off): Stage 0 ~86 GB/GPU, Stage 1 ~63 GB/GPU.
Stage 2 adds a frozen teacher and Stage 3 adds a trainable critic and a frozen teacher, so they need more memory;
neither has been run yet.

**Resume.** Rerun the same command with the same `RUN_NAME`, `OUTPUT_ROOT`, bundle and GPU count. The
Framework resumes from `latest_checkpoint.txt`.

## 4. Export and generate

```bash
# Export (HF format, ~34 GB). Use the run's own config so the export keeps that stage's attention setup.
CKPT=$RUNS/agibot-clip15-stage3/checkpoints/iter_000001500 RUN_DIR=$RUNS/agibot-clip15-stage3 STAGE=3 \
  OUT=exports/stage3 scripts/export_stage.sh

# Forward dynamics: first frame + action sequence -> video
CKPT=exports/stage3 MANIFEST=gen_inputs/manifest.jsonl OUT=gen/stage3 MODE=ar STEPS=4 GPU=0 scripts/generate.sh
```

- `MODE=full` samples the whole clip at once (Stage 0, bidirectional). `MODE=ar` samples causally chunk by
  chunk (Stages 1–3).
- `STEPS=35` for undistilled Stage 0/1 exports; a distilled Stage 3 student uses 4.
- `GPU=0,1` shards the weights over two GPUs with FSDP when one GPU lacks memory.

Manifest rows (one JSON per line) follow the Framework's bimanual inference contract:

```json
{"name": "clip", "prompt": "<caption>", "vision_path": "first_frame.png", "action_path": "actions.json",
 "model_mode": "forward_dynamics", "format_prompt_as_json": false, "action_chunk_size": 116,
 "num_frames": 117, "fps": 15, "seed": 2026}
```

where `actions.json` is `{"action_space": "normalized_unified_v1", "action": [[...59 values...], ...],
"domain_names": "agibotworld"}`. The number of actions must equal `num_frames - 1` and be divisible by 4.
`scripts/prep_clip_gen_inputs.py` writes these inputs (plus the ground-truth clip) straight from a bundle
through the Framework data loader, and `scripts/compare_clip_gen.py` reports per-frame PSNR against the real
clip and writes side-by-side videos.

## Repository layout

```
scripts/
  build_clip_bundle_full.py   full 15 fps bundle from the raw sources (resumable, parallel)
  pack_raw_sources.py         how the Hugging Face tars were made
  build_clip_bundle.py        same conversion for a hand-picked list of clips (small test bundles)
  build_bimanual_bundle.py    shared ACT2 / bundle writer (also builds 30 fps bundles from raw Beta episodes)
  validate_bundle.sh          Framework static check + per-stage materialization
  train_stage.sh              one stage on one node (default 4 GPUs, global batch 16)
  export_stage.sh             training checkpoint -> HF inference checkpoint
  generate.sh                 forward-dynamics generation
  prep_clip_gen_inputs.py     generation inputs + ground truth from a bundle
  compare_clip_gen.py         PSNR / side-by-side comparison
  tests/                      the exact 2-GPU, 2-clip test launchers we ran
patches/
  0001-causal-attention-skip-short-segments.patch
```

## Licenses and credits

- Training data is derived from [AgiBot World](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)
  (CC BY-NC-SA 4.0); the dataset keeps that license.
- Cosmos Framework and the Cosmos3-Nano weights are NVIDIA's and follow their own licenses; the patch in
  `patches/` modifies Framework code.
- The clip segmentation and visual captions come from our lab's AgiBot processing pipeline.
