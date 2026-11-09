"""Build a 15 fps Sim-Bimanual bundle from wzy pipeline_output_fix/agibot action clips.

Per clip <task>_<episode>_a<k>_<start>_<end>:
  video   : pipeline_output_fix/agibot/<clip>/head/rgb.mp4   (640x480, 15 fps, clip frame i)
  state   : agibot_processed/proprio_stats/<task>/<episode>/proprio_stats.h5 row 2*(start+i)  (30 fps -> 15 fps)
  caption : <clip>/head/caption.pickle  (visual caption + action_text)
Alignment is asserted per clip against the clip's own episode_meta joint angles.
Each clip is one sample; observations trimmed to 1 + 4k. Reuses build_bimanual_bundle for ACT2, the
gripper-manifest rank mapping, sample records and bundle layout (with FPS overridden to 15).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_bimanual_bundle as bb  # noqa: E402

CLIP_ROOT = Path(os.environ.get("AGIBOT_CLIP_ROOT", "/nfs/turbo/coe-jungaocv-turbo2/wzy/datasets/pipeline_output_fix/agibot"))
FPS = 15.0
STRIDE = 2  # raw 30 fps rows per clip frame


def lerobot_rank(task: str) -> dict[str, int]:
    infos = json.loads((bb.PROC_ROOT / "task_info" / f"task_{task}.json").read_text())
    audited = bb.gripper_manifest_count(task)
    ids = sorted((str(e["episode_id"]) for e in infos), key=int)
    return {eid: r for r, eid in enumerate(ids) if r < audited}


def convert_clip(clip: str, out: Path, ab) -> dict:
    task, episode, _a, start, end = clip.split("_")
    start, end = int(start), int(end)
    head = CLIP_ROOT / clip / "head"
    video = head / "rgb.mp4"
    nv, w, h, fps = bb.video_frames(video)
    if (w, h) != (640, 480) or abs(fps - FPS) > 1e-3 or nv != end - start:
        raise ValueError(f"unexpected clip video {w}x{h}@{fps} frames={nv} (expected {end - start})")
    fields, n, notes = bb.build_act2_fields(bb.PROC_ROOT / "proprio_stats" / task / episode / "proprio_stats.h5")
    rows = STRIDE * (start + np.arange(nv))
    if rows[-1] >= n:
        raise ValueError(f"clip exceeds proprio ({rows[-1]} >= {n})")
    fields = {k: v[rows] for k, v in fields.items()}
    meta = np.load(head / "episode_meta.npz")
    ja = np.concatenate([meta["joint_angles_left"], meta["joint_angles_right"]], axis=1)
    err = float(np.abs(fields["state_joint"] - ja).max())
    if err > 1e-4:
        raise ValueError(f"proprio/clip joint misalignment {err}")
    cap = pickle.load(open(head / "caption.pickle", "rb"))
    obs = 1 + 4 * ((nv - 1) // 4)
    caption = f"{cap['caption'].strip()} [0.0s-{(obs - 1) / FPS:.1f}s]: {cap['action_text'].strip()}"

    base = f"local://agibot-clip15/{clip}"
    act_uri, vid_uri = f"{base}/actions.bin", f"{base}/video.mp4"
    act_rel, vid_rel = bb.asset_rel(act_uri, "bin"), bb.asset_rel(vid_uri, "mp4")
    sdir = out / "train"
    for rel in (act_rel, vid_rel):
        (sdir / rel).parent.mkdir(parents=True, exist_ok=True)
    payload = ab.encode_action_bin({k: v.astype(np.float32) for k, v in fields.items()}, bb.SOURCE_DATASET)
    assert ab.decode_action_bin(payload)["state_joint"].shape == (nv, 14)
    (sdir / act_rel).write_bytes(payload)
    shutil.copyfile(video, sdir / vid_rel)
    return {"task": task, "episode": episode, "clip": clip, "status": "ok", "split": "train", "frames": nv,
            "notes": notes + [f"joint_align_max_err:{err:.2e}"], "nb_bytes": (sdir / vid_rel).stat().st_size,
            "assets": {act_uri: act_rel, vid_uri: vid_rel},
            "windows": [{"start": 0, "obs": obs, "caption": caption}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    bb.FPS = FPS  # sample records: source/target fps and framerate
    ab = bb._load_action_binary()
    eps = [convert_clip(c, out, ab) for c in args.clips]
    index = {}
    for ep in eps:
        index[(ep["task"], ep["episode"])] = lerobot_rank(ep["task"])[ep["episode"]]
    records = bb.build_records(eps, index)
    for r, ep in zip(records, eps):
        r["identity"] = f"agibot_clip15_{ep['clip']}"
    max_obs = max(ep["windows"][0]["obs"] for ep in eps)
    bb.write_bundle(out / "train", Path(args.template), records, eps,
                    SimpleNamespace(window_obs=max_obs, min_obs=0, val_frac=0.0))
    rec_path = out / "train" / "receipts" / "dataset-receipt.json"
    rec = json.loads(rec_path.read_text())
    rec.update(fps=FPS, resampling="clip video is native 15 fps; proprio rows 2*(start+i) of the 30 fps record",
               sources={"video_and_captions": str(CLIP_ROOT), "proprio": str(bb.PROC_ROOT / "proprio_stats")},
               window_policy="one sample per action clip, observations trimmed to 1+4k",
               caption_policy="pipeline_output_fix visual caption + clip action_text",
               clips=args.clips, exporter="cosmos3_vllm_omni/scripts/build_clip_bundle.py")
    rec_path.write_text(json.dumps(rec, indent=1))
    print(json.dumps([{k: ep[k] for k in ("clip", "frames", "notes")} | {"obs": ep["windows"][0]["obs"],
                      "caption": ep["windows"][0]["caption"]} for ep in eps], indent=1))


if __name__ == "__main__":
    main()
