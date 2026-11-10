"""Build the full 15 fps Sim-Bimanual bundle from every usable pipeline_output_fix/agibot action clip.

Scales build_clip_bundle.py (same per-clip conversion) to the whole train_manifest.json:
  - clips are grouped by episode; each episode's proprio h5 is read once,
  - episodes go to train/val by sha256(task/episode) (every clip of an episode stays in one split),
  - per-episode results are cached in <out>/_work/episodes/ so a rerun resumes,
  - failures are recorded in <out>/_work/skipped.json, not fatal.
Each clip is one sample (observations trimmed to 1 + 4k, minimum --min-obs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_bimanual_bundle as bb  # noqa: E402

CLIP_ROOT = Path(os.environ.get("AGIBOT_CLIP_ROOT", "/nfs/turbo/coe-jungaocv-turbo2/wzy/datasets/pipeline_output_fix/agibot"))
FPS = 15.0
STRIDE = 2  # raw 30 fps proprio rows per 15 fps clip frame


def task_ranks(task: str) -> dict[str, int]:
    """Raw episode id -> per-task LeRobot episode index (the key of the audited gripper manifest)."""
    infos = json.loads((bb.PROC_ROOT / "task_info" / f"task_{task}.json").read_text())
    audited = bb.gripper_manifest_count(task)
    ids = sorted((str(e["episode_id"]) for e in infos), key=int)
    return {eid: r for r, eid in enumerate(ids) if r < audited}


def convert_episode(task: str, episode: str, clips: list[str], split: str, out: Path, min_obs: int) -> dict:
    cache = out / "_work" / "episodes" / f"{task}_{episode}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    ab = bb._load_action_binary()
    res: dict = {"task": task, "episode": episode, "split": split, "clips": [], "skipped": []}
    try:
        fields_all, n, notes = bb.build_act2_fields(bb.PROC_ROOT / "proprio_stats" / task / episode / "proprio_stats.h5")
    except Exception as exc:
        res["skipped"] = [{"clip": c, "reason": f"proprio: {type(exc).__name__}: {exc}"} for c in clips]
        clips = []
    for clip in clips:
        try:
            _t, _e, _a, start, end = clip.split("_")
            start, end = int(start), int(end)
            head = CLIP_ROOT / clip / "head"
            video = head / "rgb.mp4"
            nv, w, h, fps = bb.video_frames(video)
            if (w, h) != (640, 480) or abs(fps - FPS) > 1e-3 or nv != end - start:
                raise ValueError(f"unexpected clip video {w}x{h}@{fps} frames={nv} (expected {end - start})")
            obs = 1 + 4 * ((nv - 1) // 4)
            if obs < min_obs:
                raise ValueError(f"too short ({nv} frames)")
            rows = STRIDE * (start + np.arange(nv))
            if rows[-1] >= n:
                raise ValueError(f"clip exceeds proprio ({rows[-1]} >= {n})")
            fields = {k: v[rows] for k, v in fields_all.items()}
            with open(head / "caption.pickle", "rb") as f:
                cap = pickle.load(f)
            caption = f"{cap['caption'].strip()} [0.0s-{(obs - 1) / FPS:.1f}s]: {cap['action_text'].strip()}"
            base = f"local://agibot-clip15/{clip}"
            act_uri, vid_uri = f"{base}/actions.bin", f"{base}/video.mp4"
            act_rel, vid_rel = bb.asset_rel(act_uri, "bin"), bb.asset_rel(vid_uri, "mp4")
            sdir = out / split
            for rel in (act_rel, vid_rel):
                (sdir / rel).parent.mkdir(parents=True, exist_ok=True)
            payload = ab.encode_action_bin({k: v.astype(np.float32) for k, v in fields.items()}, bb.SOURCE_DATASET)
            assert ab.decode_action_bin(payload)["state_joint"].shape == (nv, 14)
            (sdir / (act_rel + ".part")).write_bytes(payload)
            (sdir / (act_rel + ".part")).rename(sdir / act_rel)
            shutil.copyfile(video, sdir / (vid_rel + ".part"))
            (sdir / (vid_rel + ".part")).rename(sdir / vid_rel)
            res["clips"].append({
                "task": task, "episode": episode, "clip": clip, "status": "ok", "split": split, "frames": nv,
                "notes": notes, "nb_bytes": (sdir / vid_rel).stat().st_size,
                "assets": {act_uri: act_rel, vid_uri: vid_rel},
                "windows": [{"start": 0, "obs": obs, "caption": caption}]})
        except Exception as exc:  # recorded, not fatal
            res["skipped"].append({"clip": clip, "reason": f"{type(exc).__name__}: {exc}"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


def clip_records(eps: list[dict], ranks: dict[tuple[str, str], int]) -> list[dict]:
    records = bb.build_records(eps, ranks)
    for r, ep in zip(records, eps):
        clip, obs = ep["clip"], ep["windows"][0]["obs"]
        r["identity"] = f"agibot_clip15_{clip}"
        r["request"]["aux_seed"] = int(hashlib.sha256(f"clip15/{clip}".encode()).hexdigest()[:15], 16)
        # Unique per clip; the gripper lookup only reads the .../task_<t>/episode_<rank> prefix.
        row = r["source_rows"][bb.VIEW]
        prefix = row["__canonical_source_sample_id__"].rpartition("/")[0]
        row["__canonical_source_sample_id__"] = f"{prefix}/clip_{clip}:0:{obs}"
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", required=True, help="extracted HF example bundle directory")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--min-obs", type=int, default=33, help="minimum observations per clip (33 = ~2.1 s)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-episodes", type=int, default=None, help="smoke tests")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "_work").mkdir(parents=True, exist_ok=True)
    bb.FPS = FPS  # sample records: source/target fps and framerate

    kept = json.loads((CLIP_ROOT / "train_manifest.json").read_text())["kept"]
    by_ep: dict[tuple[str, str], list[str]] = defaultdict(list)
    for clip in kept:
        task, episode = clip.split("_")[:2]
        by_ep[(task, episode)].append(clip)
    ranks: dict[tuple[str, str], int] = {}
    unmapped = []
    for task in sorted({t for t, _ in by_ep}):
        try:
            tr = task_ranks(task)
        except Exception as exc:
            tr = {}
            unmapped.append(f"task {task}: {type(exc).__name__}: {exc}")
        for (t, e) in by_ep:
            if t == task and e in tr:
                ranks[(t, e)] = tr[e]
    jobs = []
    for (task, episode), clips in by_ep.items():
        if (task, episode) not in ranks:
            unmapped.append(f"{task}/{episode}: no audited gripper rank ({len(clips)} clips)")
        elif not (bb.PROC_ROOT / "proprio_stats" / task / episode / "proprio_stats.h5").exists():
            unmapped.append(f"{task}/{episode}: no proprio h5 ({len(clips)} clips)")
        else:
            jobs.append((task, episode, sorted(clips), bb.split_of(task, episode, args.val_frac)))
    (out / "_work" / "unmapped.txt").write_text("\n".join(unmapped) + "\n")
    jobs.sort(key=lambda j: hashlib.sha256(f"{j[0]}/{j[1]}".encode()).hexdigest())  # task-mixed order
    if args.limit_episodes:
        jobs = jobs[: args.limit_episodes]
    print(f"{len(jobs)} episodes / {sum(len(j[2]) for j in jobs)} clips to process "
          f"({len(unmapped)} episodes/tasks unmapped)", flush=True)

    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(convert_episode, t, e, c, s, out, args.min_obs) for t, e, c, s in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if k % 500 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} episodes", flush=True)

    skipped = [s | {"episode": f"{r['task']}/{r['episode']}"} for r in results for s in r["skipped"]]
    summary = {"clips_ok": sum(len(r["clips"]) for r in results), "clips_skipped": len(skipped), "splits": {}}
    for split in ("train", "val"):
        eps = sorted((c for r in results if r["split"] == split for c in r["clips"]), key=lambda c: c["clip"])
        if not eps:
            continue
        records = clip_records(eps, ranks)
        max_obs = max(c["windows"][0]["obs"] for c in eps)
        bb.write_bundle(out / split, Path(args.template), records, eps,
                        SimpleNamespace(window_obs=max_obs, min_obs=args.min_obs, val_frac=args.val_frac))
        rec_path = out / split / "receipts" / "dataset-receipt.json"
        rec = json.loads(rec_path.read_text())
        rec.update(fps=FPS, resampling="clip video is native 15 fps; proprio rows 2*(start+i) of the 30 fps record",
                   sources={"video_and_captions": str(CLIP_ROOT), "proprio": str(bb.PROC_ROOT / "proprio_stats")},
                   window_policy=f"one sample per action clip, observations trimmed to 1+4k (min {args.min_obs})",
                   caption_policy="pipeline_output_fix visual caption + clip action_text",
                   episodes=len({(c["task"], c["episode"]) for c in eps}), samples=len(records),
                   exporter="agibot-cosmos3-distill/scripts/build_clip_bundle_full.py")
        rec_path.write_text(json.dumps(rec, indent=1))
        summary["splits"][split] = {
            "episodes": rec["episodes"], "clips": len(records), "tasks": len({c["task"] for c in eps}),
            "hours": round(sum(c["windows"][0]["obs"] - 1 for c in eps) / FPS / 3600, 2)}
    (out / "_work" / "skipped.json").write_text(json.dumps(skipped, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
