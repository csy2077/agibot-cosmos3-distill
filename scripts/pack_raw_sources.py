"""Pack the raw sources of the 15 fps bundle into per-task tar files for the Hugging Face dataset.

Reads (never modifies) the clip pipeline output and the processed Beta records; writes to --out:
  clips/task_<t>.tar          clips/<clip>/head/{rgb.mp4, caption.pickle, episode_meta.npz}
  proprio_stats/task_<t>.tar  proprio_stats/<t>/<episode>/proprio_stats.h5   (only episodes that have clips)
  task_info/task_<t>.json
  clips_train_manifest.json   (copy of the clip pipeline's train_manifest.json)
Extracting every tar into one directory DATA gives DATA/clips/... and DATA/proprio_stats/..., i.e.
AGIBOT_CLIP_ROOT=DATA/clips and AGIBOT_PROCESSED_ROOT=DATA for the bundle builders
(copy clips_train_manifest.json to DATA/clips/train_manifest.json).
Resumable: finished tars are skipped; partial ones are written as .part and renamed when complete.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CLIP_FILES = ("rgb.mp4", "caption.pickle", "episode_meta.npz")


def pack(dst: Path, members: list[tuple[Path, str]]) -> tuple[str, int, int]:
    if dst.exists():
        return dst.name, -1, 0
    part = dst.with_suffix(".tar.part")
    n = missing = 0
    with tarfile.open(part, "w", format=tarfile.PAX_FORMAT) as tar:
        for src, arc in members:
            if src.exists():
                tar.add(src, arcname=arc, recursive=False)
                n += 1
            else:
                missing += 1
    part.rename(dst)
    return dst.name, n, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-root", default="/nfs/turbo/coe-jungaocv-turbo2/wzy/datasets/pipeline_output_fix/agibot")
    ap.add_argument("--processed-root", default="/nfs/turbo/coe-jungaocv-turbo2/wzy/datasets/agibot_processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    clip_root, proc, out = Path(args.clip_root), Path(args.processed_root), Path(args.out)
    for d in ("clips", "proprio_stats", "task_info"):
        (out / d).mkdir(parents=True, exist_ok=True)

    kept = json.loads((clip_root / "train_manifest.json").read_text())["kept"]
    clips_by_task: dict[str, list[str]] = defaultdict(list)
    eps_by_task: dict[str, set[str]] = defaultdict(set)
    for clip in kept:
        task, episode = clip.split("_")[:2]
        clips_by_task[task].append(clip)
        eps_by_task[task].add(episode)
    shutil.copyfile(clip_root / "train_manifest.json", out / "clips_train_manifest.json")
    for task in clips_by_task:
        src = proc / "task_info" / f"task_{task}.json"
        if src.exists() and not (out / "task_info" / src.name).exists():
            shutil.copyfile(src, out / "task_info" / src.name)

    jobs = []
    for task in sorted(clips_by_task, key=int):
        jobs.append((out / "clips" / f"task_{task}.tar",
                     [(clip_root / c / "head" / f, f"clips/{c}/head/{f}") for c in sorted(clips_by_task[task])
                      for f in CLIP_FILES]))
        jobs.append((out / "proprio_stats" / f"task_{task}.tar",
                     [(proc / "proprio_stats" / task / e / "proprio_stats.h5", f"proprio_stats/{task}/{e}/proprio_stats.h5")
                      for e in sorted(eps_by_task[task], key=int)]))
    print(f"{len(clips_by_task)} tasks, {len(kept)} clips, {sum(map(len, eps_by_task.values()))} episodes, "
          f"{len(jobs)} tars", flush=True)
    report = {}
    with ThreadPoolExecutor(args.workers) as pool:  # I/O bound: threads are enough
        futs = [pool.submit(pack, dst, members) for dst, members in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            name, n, missing = fut.result()
            report[name] = {"files": n, "missing": missing}
            print(f"  [{k}/{len(jobs)}] {name}: {'already done' if n < 0 else f'{n} files, {missing} missing'}",
                  flush=True)
    (out / "pack_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
