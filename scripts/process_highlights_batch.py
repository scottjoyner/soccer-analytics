"""Batch process permitted local highlight videos into per-match YOLO detection tables.

The source footage is AV1-encoded; the local media reader has no AV1 decoder, so each
file is first transcoded to H.264 (ffmpeg) and then run through detect-yolo. Detection
tables are written to data/processed/highlights/detections/<match_id>/detections.parquet.

Resumable: files already processed (detections.parquet present) are skipped.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

from soccer_edge.video.yolo_pipeline import run_yolo_detection

SRC = Path("/media/scott/SSD_4TB/fileserver/highlights")
MODEL = Path("/home/scott/git/soccer-analytics/yolov8n.pt")
TRANSCODE_ROOT = Path("/tmp/hl/transcoded")
OUT_ROOT = Path("/home/scott/git/soccer-analytics/data/processed/highlights/detections")
STRIDE = 3
MAX_SAMPLES = 150
CONF = 0.30
FPS = 2


def match_id_for(path: Path) -> str:
    m = re.match(r"\s*(\d+)", path.stem)
    return f"HL{m.group(1)}" if m else path.stem.replace(" ", "_")


def transcode(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", "0", "-i", str(src),
        "-vf", f"fps={FPS}", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    files = sorted(SRC.glob("*.mp4"))
    total = len(files)
    TRANSCODE_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, src in enumerate(files, 1):
        mid = match_id_for(src)
        trans = TRANSCODE_ROOT / f"{mid}.mp4"
        outdir = OUT_ROOT / mid
        det = outdir / "detections.parquet"
        if det.exists():
            done += 1
            continue
        if not trans.exists():
            transcode(src, trans)
        outdir.mkdir(parents=True, exist_ok=True)
        # These clips are a known, rights-approved local source (pre-approved by the
        # operator, not scraped/remote). The CLI gate is bypassed deliberately here via
        # enforce_rights=False; a production run should instead pass a manifest row so
        # every clip is gated. The base run_yolo_detection now refuses un-gated footage.
        run_yolo_detection(
            input_path=trans, output_dir=outdir, model_path=MODEL,
            stride=STRIDE, max_samples=MAX_SAMPLES, confidence_threshold=CONF, transform=None,
            enforce_rights=False,
        )
        done += 1
        print(f"[{done}/{total}] {mid} -> {det}", flush=True)
    print(f"COMPLETE: {done}/{total} matches processed", flush=True)


if __name__ == "__main__":
    main()
