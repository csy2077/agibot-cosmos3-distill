"""Write Bimanual forward-dynamics inference rows for clip-bundle samples, using the native loader's
normalized unified_v1 actions (exactly what training saw), the clip's first frame, and a GT clip."""
import json, subprocess, sys
from pathlib import Path
import torch
from cosmos_framework.data.generator.local_simulation.bimanual_config import make_bimanual_local_dataloader

bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
clip_root = Path("/nfs/turbo/coe-jungaocv-turbo2/wzy/datasets/pipeline_output_fix/agibot")
samples = {s["identity"]: s for s in json.loads((bundle / "samples.json").read_text())}
loader = make_bimanual_local_dataloader(bundle_path=str(bundle), stage=1, tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4, patch_spatial=2, max_sequence_length=None, max_samples_per_batch=1,
    num_workers=0, seed=42)
seen, rows = set(), []
for step, b in enumerate(loader):
    sid = b["local_sample_id"][0]; print("step", step, sid, flush=True)
    if len(seen) == len(samples) or step > 50:
        break
    if sid in seen:
        continue
    seen.add(sid)
    a = b["action"][0]; a = (a[0] if isinstance(a, (list, tuple)) else a).float()
    dom = int(b["domain_id"][0]); fps = float(b["conditioning_fps"][0].item())
    assert dom == 15, f"unexpected domain {dom}"
    assert torch.all(a[:, 59:] == 0)
    clip = sid.removeprefix("agibot_clip15_")
    n = a.shape[0]
    d = out / clip
    d.mkdir(parents=True, exist_ok=True)
    (d / "actions_normalized_unifiedv1.json").write_text(json.dumps(
        {"action_space": "normalized_unified_v1", "action": a[:, :59].tolist(), "domain_names": "agibotworld"}))
    src = clip_root / clip / "head" / "rgb.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-frames:v", "1", str(d / "first_frame.png")], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-frames:v", str(n + 1), "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(d / "gt.mp4")], check=True)
    rows.append({"name": clip, "model_mode": "forward_dynamics", "prompt": b["ai_caption"][0],
                 "format_prompt_as_json": False, "vision_path": str(d / "first_frame.png"),
                 "action_path": str(d / "actions_normalized_unifiedv1.json"), "domain_name": "agibotworld",
                 "action_chunk_size": n, "num_frames": n + 1, "fps": int(fps), "seed": 2026})
    print(clip, "actions", tuple(a.shape), "fps", fps, "domain", dom, "|a| mean", round(a[:, :59].abs().mean().item(), 3))
(out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
