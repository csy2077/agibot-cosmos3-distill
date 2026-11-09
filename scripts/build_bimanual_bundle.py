"""Build Cosmos3-Nano-Sim-Bimanual local training bundles from AgiBotWorld-Beta raw data.

Sources (per episode <task>/<episode>):
  video     : yizhong/data/agibot_beta/<task>/<episode>/head_color.mp4      (640x480, 30 fps)
  proprio   : wzy/datasets/agibot_processed/proprio_stats/<task>/<episode>/proprio_stats.h5
  captions  : wzy/datasets/agibot_processed/task_info/task_<task>.json      (task, scene, 30 fps segments)

Output mirrors the published HF example bundle (examples/training/sim_bimanual_robot_example_v1):
  <out>/<split>/{manifest.json, samples.json, assets/, resources/{global_asinh.json, tokenizer/}, receipts/}
ACT2 payloads are written with the Framework's own `encode_action_bin` (agibot_world spec), whose
field packing is documented in cosmos_framework/data/generator/action/action_binary.py.

Each episode is stored once (full video + full ACT2); windows reference it via source_frame_indices.
Resumable: per-episode results are cached in <out>/_work/episodes/<task>_<episode>.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

FW = Path(os.environ.get("COSMOS_FRAMEWORK_DIR", "/nfs/turbo/coe-jungaocv-turbo2/siyiche/cosmos/cosmos-framework"))
VIDEO_ROOT = Path(os.environ.get("AGIBOT_BETA_VIDEO_ROOT", "/nfs/turbo/coe-jungaocv-turbo2/yizhong/data/agibot_beta"))
PROC_ROOT = Path(os.environ.get("AGIBOT_PROCESSED_ROOT", "/nfs/turbo/coe-jungaocv-turbo2/wzy/datasets/agibot_processed"))
SOURCE_DATASET = "agibot_world__AgiBotWorld-Beta_20260102"
DATASET_ID = "agibotworld_beta_20260102"
VIEW = "observation.images.head"
FPS = 30.0
GLOBAL_ASINH_SHA256 = "ae695e2c643a5626b3c3564f00705aa3a83ab0da7640549c7340db71172594b0"
NS = uuid.UUID("6f1c3b52-6a43-4f0e-9a4e-1f7f0b6a2c11")


def _load_action_binary():
    spec = importlib.util.spec_from_file_location(
        "cosmos_action_binary", FW / "cosmos_framework/data/generator/action/action_binary.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cosmos_action_binary"] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def asset_rel(uri: str, ext: str) -> str:
    h = hashlib.sha256(uri.encode()).hexdigest()
    return f"assets/{h[:2]}/{h}.{ext}"


def video_frames(path: Path) -> tuple[int, int, int, float]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(path)]
    )
    s = json.loads(out)["streams"][0]
    num, den = map(float, s["r_frame_rate"].split("/"))
    return int(s["nb_read_packets"]), int(s["width"]), int(s["height"]), num / den


def _col(f: h5py.File, key: str, n: int, width: int, notes: list[str]) -> np.ndarray:
    """Read an h5 field as [n, width]; zero-fill (and note) when absent or misaligned."""
    if key in f and f[key].shape and f[key].shape[0] == n:
        arr = np.asarray(f[key][:], dtype=np.float64).reshape(n, -1)
        if arr.shape[1] == width:
            return arr
    notes.append(f"zero_filled:{key}")
    return np.zeros((n, width))


def build_act2_fields(h5_path: Path) -> tuple[dict[str, np.ndarray], int, list[str]]:
    notes: list[str] = []
    with h5py.File(h5_path, "r") as f:
        n = int(f["timestamp"].shape[0])
        state_end_effector = np.concatenate(
            [_col(f, "state/effector/position", n, 2, notes), _col(f, "state/end/position", n, 6, notes),
             _col(f, "state/end/orientation", n, 8, notes), _col(f, "state/head/position", n, 2, notes),
             _col(f, "state/waist/position", n, 2, notes)], axis=1)
        action_end_effector = np.concatenate(
            [_col(f, "action/effector/position", n, 2, notes), _col(f, "action/end/position", n, 6, notes),
             _col(f, "action/end/orientation", n, 8, notes), _col(f, "action/head/position", n, 2, notes),
             _col(f, "action/waist/position", n, 2, notes), _col(f, "action/robot/velocity", n, 2, notes)], axis=1)
        robot_pos = _col(f, "state/robot/position", n, 3, notes)
        robot_quat = _col(f, "state/robot/orientation", n, 4, notes)
        fields = {
            "action_joint": _col(f, "action/joint/position", n, 14, notes),
            "action_end_effector": action_end_effector,
            "state_joint": _col(f, "state/joint/position", n, 14, notes),
            "state_joint_current": _col(f, "state/joint/current_value", n, 14, notes),
            "state_end_effector": state_end_effector,
        }
    # Beta stores an all-zero quaternion for a static base; write the identity (xyzw) explicitly.
    zero_q = np.linalg.norm(robot_quat, axis=1) < 1e-8
    if zero_q.any():
        robot_quat[zero_q] = [0.0, 0.0, 0.0, 1.0]
        notes.append(f"identity_base_quat_rows:{int(zero_q.sum())}")
    fields["state_robot"] = np.concatenate([robot_pos, robot_quat], axis=1)
    fk_inputs = ("state/joint/position", "state/effector/position", "state/head/position", "state/waist/position")
    if any(note == f"zero_filled:{key}" for key in fk_inputs for note in notes):  # FK inputs must be real
        raise ValueError(f"missing FK state field ({notes})")
    for name, arr in fields.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite values in {name}")
    return fields, n, notes


def gripper_manifest_count(task: str) -> int:
    """Number of audited episodes for this Beta task in the Framework gripper manifest (0 if absent)."""
    m = json.loads((FW / "cosmos_framework/data/generator/action/manifests/agibot_episode_grippers.json").read_text())
    shard = m["datasets"][SOURCE_DATASET]["shards"].get(f"task_{task}", {})
    indices = [i for group in shard.values() for i in group]
    assert sorted(indices) == list(range(len(indices))), "expected contiguous per-task indices"
    assert set(shard) <= {"robotiq"}, f"task {task} has non-Robotiq episodes; rank mapping would be unsafe"
    return len(indices)


def window_starts(n: int, window_obs: int, min_obs: int) -> list[tuple[int, int]]:
    """Non-overlapping [start, start+obs) windows; a short episode gets one trimmed window (T % 4 == 0)."""
    if n >= window_obs:
        return [(s, window_obs) for s in range(0, n - window_obs + 1, window_obs - 1)]
    t = ((n - 1) // 4) * 4
    return [(0, t + 1)] if t + 1 >= min_obs else []


def caption_for(info: dict, start: int, obs: int) -> str:
    end = start + obs - 1
    parts = [info["task_name"].strip().rstrip(".") + "."]
    if start == 0 and info.get("init_scene_text"):
        parts.append(info["init_scene_text"].strip())
    for seg in (info.get("label_info") or {}).get("action_config", []) or []:
        a, b = int(seg["start_frame"]), int(seg["end_frame"])
        if b <= start or a >= end:
            continue
        a, b = max(a, start) - start, min(b, end) - start
        parts.append(f"[{a / FPS:.1f}s-{b / FPS:.1f}s]: {seg['action_text'].strip()}")
    return " ".join(parts)


def split_of(task: str, episode: str, val_frac: float) -> str:
    h = int(hashlib.sha256(f"split-v1:{task}/{episode}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def process_episode(task: str, episode: str, info: dict, out: Path, args_d: dict) -> dict:
    cache = out / "_work" / "episodes" / f"{task}_{episode}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    ab = _load_action_binary()
    result: dict = {"task": task, "episode": episode, "status": "skipped"}
    try:
        video = VIDEO_ROOT / task / episode / "head_color.mp4"
        h5p = PROC_ROOT / "proprio_stats" / task / episode / "proprio_stats.h5"
        if not video.exists() or not h5p.exists():
            raise FileNotFoundError("missing video or proprio h5")
        fields, n, notes = build_act2_fields(h5p)
        nv, w, h, fps = video_frames(video)
        if nv != n:
            raise ValueError(f"video frames {nv} != state frames {n}")
        if (w, h) != (640, 480) or abs(fps - FPS) > 1e-3:
            raise ValueError(f"unexpected video {w}x{h}@{fps}")
        wins = window_starts(n, args_d["window_obs"], args_d["min_obs"])
        if not wins:
            raise ValueError(f"episode too short ({n} frames)")
        split = split_of(task, episode, args_d["val_frac"])
        base = f"local://agibot-beta/{task}/{episode}"
        act_uri, vid_uri = f"{base}/actions.bin", f"{base}/video.mp4"
        act_rel, vid_rel = asset_rel(act_uri, "bin"), asset_rel(vid_uri, "mp4")
        sdir = out / split
        (sdir / act_rel).parent.mkdir(parents=True, exist_ok=True)
        (sdir / vid_rel).parent.mkdir(parents=True, exist_ok=True)
        payload = ab.encode_action_bin({k: v.astype(np.float32) for k, v in fields.items()}, SOURCE_DATASET)
        assert ab.decode_action_bin(payload)["state_joint"].shape == (n, 14)
        tmp = sdir / (act_rel + ".part")
        tmp.write_bytes(payload)
        tmp.rename(sdir / act_rel)
        tmp = sdir / (vid_rel + ".part")
        shutil.copyfile(video, tmp)
        tmp.rename(sdir / vid_rel)
        result = {
            "task": task, "episode": episode, "status": "ok", "split": split, "frames": n, "notes": notes,
            "nb_bytes": (sdir / vid_rel).stat().st_size,
            "assets": {act_uri: act_rel, vid_uri: vid_rel},
            "windows": [{"start": s, "obs": o, "caption": caption_for(info, s, o)} for s, o in wins],
        }
    except Exception as exc:  # recorded, not fatal
        result = {"task": task, "episode": episode, "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


def build_records(episodes: list[dict], lerobot_index: dict[tuple[str, str], int]) -> list[dict]:
    windows = [(ep, w) for ep in episodes for w in ep["windows"]]
    total = len(windows)
    fingerprint = hashlib.sha256(
        json.dumps([[ep["task"], ep["episode"], w["start"], w["obs"]] for ep, w in windows]).encode()
    ).hexdigest()
    records = []
    for i, (ep, w) in enumerate(windows):
        task, episode, s, o = ep["task"], ep["episode"], w["start"], w["obs"]
        shard = str(uuid.uuid5(NS, f"{task}/{episode}"))
        act_uri, vid_uri = sorted(ep["assets"])  # actions.bin < video.mp4
        ep_index = int(episode)
        row_id = int(hashlib.sha256(f"{task}/{episode}".encode()).hexdigest()[:12], 16)
        records.append({
            "assets": ep["assets"],
            "identity": f"agibot_beta_{task}_{episode}_{s:06d}_{s + o - 1:06d}",
            "request": {
                "aux_seed": int(hashlib.sha256(f"{task}/{episode}/{s}".encode()).hexdigest()[:15], 16),
                "caption": w["caption"],
                "caption_mode": "short",
                "dataset_length": total,
                "dense_caption_action_sample_row_id": None,
                "dense_caption_chunk_end_source_frame": None,
                "dense_caption_chunk_start_source_frame": None,
                "draw_count": 0,
                "episode_index": ep_index,
                "frame_count": ep["frames"],
                "global_index": i,
                "index_space_fingerprint": fingerprint,
                "location": {"local_start_frame": s, "row_id": i + 1, "row_position": 0, "video_frame_count": None},
                "resolution": "480",
                "row_ids_by_view": {VIEW: row_id},
                "shard_id": shard,
                "source_fps": FPS,
                "source_frame_indices": list(range(s, s + o)),
                "source_index_version": 1,
                "source_version": 1,
                "target_fps": FPS,
            },
            "source_rows": {VIEW: {
                "__canonical_dataset_id__": DATASET_ID,
                "__canonical_source_sample_id__":
                    # <suffix>/task_<t>/episode_<per-task index>/...: the key the audited gripper manifest uses.
                    f"AgiBotWorld-Beta_20260102/agibotworld/task_{task}/episode_{lerobot_index[(task, episode)]}"
                    f"/raw_{episode}:{s}:{s + o}",
                "action_data_url": act_uri,
                "clip_url": vid_uri,
                "episode_index": ep_index,
                "framerate": FPS,
                "height": 480,
                "nb_bytes": ep["nb_bytes"],
                "nb_frames": ep["frames"],
                "shard_id": shard,
                "source_dataset": SOURCE_DATASET,
                "view_name": VIEW,
                "width": 640,
            }},
            "stratum": "agibot",
        })
    return records


def write_bundle(sdir: Path, template: Path, records: list[dict], episodes: list[dict], args) -> None:
    manifest = json.loads((template / "manifest.json").read_text())
    n = len(records)
    manifest.update(sample_count=n, target_sample_count=n, quotas={"agibot": n},
                    experiment=f"agibot_beta_bimanual_{sdir.name}", download_transport="local conversion")
    entry_ds = manifest["transforms"]["agibot"]["entry"]["dataset"]
    entry_ds["lance_dataset_spec"]["fields"]["expected_sample_count"] = n
    entry_ds["lance_dataset_spec"]["fields"]["lance_uri"] = f"offline://agibot-beta-{sdir.name}/samples.lance"
    entry_ds["lance_dataset_spec"]["fields"]["source_index_uri"] = f"offline://agibot-beta-{sdir.name}/samples.lance"
    entry_ds["chunk_length"] = args.window_obs - 1
    (sdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (sdir / "samples.json").write_text(json.dumps(records))
    res = sdir / "resources"
    if res.exists():
        shutil.rmtree(res)
    shutil.copytree(template / "resources", res)
    assert sha256_file(res / "global_asinh.json") == GLOBAL_ASINH_SHA256, "pinned global_asinh digest mismatch"
    receipts = sdir / "receipts"
    receipts.mkdir(exist_ok=True)
    shutil.copyfile(FW / ".agents/skills/cosmos3-nano-sim-bimanual/post-training/references/unified-v1-action-schema-receipt.json",
                    receipts / "unified-v1-action-schema-receipt.json")
    (receipts / "dataset-receipt.json").write_text(json.dumps({
        "dataset": "AgiBotWorld-Beta (local copy)", "dataset_id": DATASET_ID, "split": sdir.name,
        "split_rule": f"sha256(task/episode) < {args.val_frac} -> val; every window of an episode stays in one split",
        "sources": {"video": str(VIDEO_ROOT), "proprio": str(PROC_ROOT / "proprio_stats"),
                    "captions": str(PROC_ROOT / "task_info")},
        "view": VIEW, "fps": FPS, "resampling": "none (native 30 fps)", "resolution": "640x480 native",
        "window_policy": f"non-overlapping {args.window_obs}-observation windows; shorter episodes: one window "
                         f"trimmed so transitions % 4 == 0 (min {args.min_obs} observations)",
        "frame_action_alignment": "proprio row i <-> video frame i (counts verified equal per episode)",
        "act2_mapping": "action_binary.py agibot_world spec; written with encode_action_bin",
        "base_pose": "all-zero Beta base quaternions written as identity xyzw",
        "gripper_calibration": "native actuator units (120 closed, 35 open); calibrated by the Framework loader",
        "caption_policy": "template: task name + initial scene (first window only) + time-stamped task_info "
                          "action segments clipped to the window; not human-reviewed",
        "normalization_statistics_sha256": GLOBAL_ASINH_SHA256,
        "template_bundle": "nvidia/Cosmos3-Nano-Sim-Bimanual examples/training/sim_bimanual_robot_example_v1",
        "episodes": len(episodes), "samples": n,
        "exporter": "cosmos3_vllm_omni/scripts/build_bimanual_bundle.py",
    }, indent=1))
    manifest_complete = json.loads((sdir / "manifest.json").read_text())
    assert manifest_complete["complete"] is True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", required=True, help="extracted HF example bundle directory")
    ap.add_argument("--tasks", nargs="+", default=["410", "428", "483", "709"])
    ap.add_argument("--limit", type=int, default=None, help="max episodes (for smoke tests)")
    ap.add_argument("--window-obs", type=int, default=901)
    ap.add_argument("--min-obs", type=int, default=241)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out, template = Path(args.out), Path(args.template)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    lerobot_index: dict[tuple[str, str], int] = {}
    for task in args.tasks:
        infos = {str(e["episode_id"]): e for e in json.loads((PROC_ROOT / "task_info" / f"task_{task}.json").read_text())}
        # Per-task LeRobot episode index = rank of the raw episode id (all four tasks are audited Robotiq).
        audited = gripper_manifest_count(task)
        for rank, eid in enumerate(sorted(infos, key=int)):
            if rank < audited:
                lerobot_index[(task, eid)] = rank
        for episode in sorted(os.listdir(VIDEO_ROOT / task)):
            if (task, episode) in lerobot_index:
                jobs.append((task, episode, infos[episode]))
            elif episode in infos:
                (out / "_work").mkdir(exist_ok=True)
                with open(out / "_work" / "not_in_gripper_manifest.txt", "a") as f:
                    f.write(f"{task}/{episode}\n")
            else:
                (out / "_work").mkdir(exist_ok=True)
                with open(out / "_work" / "no_task_info.txt", "a") as f:
                    f.write(f"{task}/{episode}\n")
    jobs.sort(key=lambda j: hashlib.sha256(f"{j[0]}/{j[1]}".encode()).hexdigest())  # task-mixed order
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} episodes to process", flush=True)

    args_d = {"window_obs": args.window_obs, "min_obs": args.min_obs, "val_frac": args.val_frac}
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(process_episode, t, e, info, out, args_d) for t, e, info in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if k % 50 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} episodes", flush=True)

    ok = sorted((r for r in results if r["status"] == "ok"), key=lambda r: (r["task"], r["episode"]))
    skipped = [r for r in results if r["status"] != "ok"]
    summary = {"episodes_ok": len(ok), "episodes_skipped": len(skipped), "splits": {}}
    for split in ("train", "val"):
        eps = [r for r in ok if r["split"] == split]
        if not eps:
            continue
        records = build_records(eps, lerobot_index)
        write_bundle(out / split, template, records, eps, args)
        summary["splits"][split] = {"episodes": len(eps), "samples": len(records),
                                    "hours": round(sum(w["obs"] - 1 for e in eps for w in e["windows"]) / FPS / 3600, 2)}
    (out / "_work" / "skipped.json").write_text(json.dumps(skipped, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
