"""Compare generated clips with ground truth: per-frame PSNR, side-by-side video and contact sheet.

Usage: compare_clip_gen.py <gen_dir> <out_dir> <run_name>[=<subdir>] ...
Each run: <gen_dir>/<subdir>/<clip...>/vision.mp4 (a run's clip dir may carry a suffix such as
'__actions_from_<other>'); ground truth: <gen_dir>/inputs/<clip>/gt.mp4.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_video(path: Path) -> np.ndarray:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    w, h = s["width"], s["height"]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 8, 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0))
    return np.asarray(im)


gen, out = Path(sys.argv[1]), Path(sys.argv[2])
runs = dict(a.split("=", 1) if "=" in a else (a, a) for a in sys.argv[3:])
out.mkdir(parents=True, exist_ok=True)
report = {}
for clip_dir in sorted((gen / "inputs").iterdir()):
    if not clip_dir.is_dir():
        continue
    clip = clip_dir.name
    gt = read_video(clip_dir / "gt.mp4")
    vids, report[clip] = {"real": gt}, {}
    for name, sub in runs.items():
        cands = sorted((gen / sub).glob(f"{clip}*/vision.mp4"))
        if not cands:
            continue
        pred = read_video(cands[0])
        n = min(len(pred), len(gt))
        p = psnr(pred[:n], gt[:n])
        vids[name] = pred[:n]
        report[clip][name] = {"frames": n, "psnr_mean": round(float(p[1:].mean()), 2),
                              "psnr_first_half": round(float(p[1:n // 2].mean()), 2),
                              "psnr_second_half": round(float(p[n // 2:].mean()), 2),
                              "psnr_frame0": round(float(p[0]), 2), "per_frame": [round(float(x), 2) for x in p]}
    n = min(len(v) for v in vids.values())
    names = list(vids)
    frames = [np.concatenate([label(vids[k][t], f"{k}  t={t / 15:.1f}s") for k in names], axis=1) for t in range(n)]
    h, w = frames[0].shape[:2]
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", "15", "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                            str(out / f"{clip}_compare.mp4")], stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    enc.wait()
    picks = np.linspace(0, n - 1, 5).round().astype(int)
    sheet = np.concatenate([frames[t] for t in picks], axis=0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / f"{clip}_sheet.jpg", quality=88)
(out / "psnr.json").write_text(json.dumps(report, indent=1))
for clip, r in report.items():
    for name, m in r.items():
        print(f"{clip:24s} {name:10s} PSNR mean {m['psnr_mean']:6.2f}  first half {m['psnr_first_half']:6.2f}  "
              f"second half {m['psnr_second_half']:6.2f}  (frame0 {m['psnr_frame0']:.1f})")
