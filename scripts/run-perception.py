#!/usr/bin/env python3
"""Run the full perception pipeline on all gameplay-cropped frames.

Outputs to datasets/perception/:
  - calibration/       visual overlays showing detected UI regions per clip
  - ocr/               extracted text from timer, scores, etc.
  - characters/        brawler blob detections + top-k portrait matches
  - summary.json       aggregate stats

Usage:
  uv run python scripts/run-perception.py
  uv run python scripts/run-perception.py --clip "batch_00_1 Hour*"
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brawlstar_agent.ui_regions import (
    COMMON_REGIONS, MODE_REGIONS, crop_normalized,
    detect_game_mode, draw_regions_overlay, norm_to_pixel,
)
from brawlstar_agent.ocr import extract_timer, extract_number, TESSERACT_AVAILABLE
from brawlstar_agent.character_match import find_similar_brawlers, detect_brawler_blobs

PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")
CROPPED_ROOT = PROJECT_ROOT / "datasets" / "gameplay_cropped"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "perception"


def process_frame(frame_path: Path, frame: np.ndarray) -> dict:
    """Run full perception on a single frame."""
    h, w = frame.shape[:2]
    result = {
        "file": frame_path.name,
        "resolution": f"{w}x{h}",
        "mode": detect_game_mode(frame),
    }

    # OCR on timer region
    timer_crop = crop_normalized(frame, COMMON_REGIONS["timer"])
    if TESSERACT_AVAILABLE:
        result["timer_ocr"] = extract_timer(timer_crop)
    else:
        result["timer_ocr"] = {"text": "", "available": False}

    # OCR on top-left info
    tl_crop = crop_normalized(frame, COMMON_REGIONS["top_left_info"])
    if TESSERACT_AVAILABLE:
        result["top_left_ocr"] = extract_number(tl_crop)
    else:
        result["top_left_ocr"] = {"text": "", "available": False}

    # Brawler blob detection in game area
    game_area = crop_normalized(frame, COMMON_REGIONS["game_area"])
    blobs = detect_brawler_blobs(game_area)
    result["brawler_blobs"] = len(blobs)

    # Character matching on first few blobs
    matches = []
    ga_x, ga_y, _, _ = norm_to_pixel(COMMON_REGIONS["game_area"], w, h)
    for bx, by, bw, bh in blobs[:5]:
        blob_crop = game_area[by:by+bh, bx:bx+bw]
        if blob_crop.size < 100:
            continue
        top_matches = find_similar_brawlers(blob_crop, top_k=3)
        matches.append({
            "bbox": [bx + ga_x, by + ga_y, bw, bh],
            "candidates": [{"name": m["name"], "dist": round(m["distance"], 3)} for m in top_matches],
        })
    result["character_matches"] = matches

    return result


def process_clip(clip_dir: Path):
    """Process all frames in a clip directory."""
    clip_name = clip_dir.name
    frames = sorted(clip_dir.glob("*.jpg"))
    if not frames:
        return

    # Output dirs
    cal_dir = OUTPUT_ROOT / "calibration"
    ocr_dir = OUTPUT_ROOT / "ocr"
    char_dir = OUTPUT_ROOT / "characters"
    for d in [cal_dir, ocr_dir, char_dir]:
        d.mkdir(parents=True, exist_ok=True)

    clip_results = []
    modes = []

    for fp in frames:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue

        result = process_frame(fp, frame)
        clip_results.append(result)
        modes.append(result["mode"])

        # Save calibration overlay for first frame of each clip
        if fp == frames[0]:
            overlay = draw_regions_overlay(frame, mode=result["mode"])
            cv2.imwrite(str(cal_dir / f"{clip_name}.jpg"), overlay)

    # Save per-clip OCR + character results
    clip_output = {
        "clip": clip_name,
        "frame_count": len(clip_results),
        "detected_modes": {str(k): int(v) for k, v in zip(*np.unique(modes, return_counts=True))},
        "tesseract_available": TESSERACT_AVAILABLE,
        "frames": clip_results,
    }

    with open(ocr_dir / f"{clip_name}.json", "w") as f:
        json.dump(clip_output, f, indent=2)

    # Summary line
    mode_str = max(set(modes), key=modes.count) if modes else "?"
    timer_texts = [r["timer_ocr"]["text"] for r in clip_results if r["timer_ocr"].get("text")]
    blob_counts = [r["brawler_blobs"] for r in clip_results]
    avg_blobs = sum(blob_counts) / len(blob_counts) if blob_counts else 0

    print(f"  {clip_name[:60]:<62} mode={mode_str:<12} "
          f"timers={len(timer_texts):>2}/{len(clip_results)}  "
          f"avg_blobs={avg_blobs:.1f}")

    return clip_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=None, help="Glob pattern to filter clips")
    args = parser.parse_args()

    clip_dirs = sorted(d for d in CROPPED_ROOT.iterdir() if d.is_dir() and list(d.glob("*.jpg")))
    if args.clip:
        import fnmatch
        clip_dirs = [d for d in clip_dirs if fnmatch.fnmatch(d.name, args.clip)]

    print(f"Running perception on {len(clip_dirs)} clips ({sum(len(list(d.glob('*.jpg'))) for d in clip_dirs)} frames)")
    print(f"Tesseract: {'YES' if TESSERACT_AVAILABLE else 'NO (OCR will be empty)'}")
    print(f"Output: {OUTPUT_ROOT}\n")

    all_results = []
    for clip_dir in clip_dirs:
        result = process_clip(clip_dir)
        if result:
            all_results.append(result)

    # Write aggregate summary
    summary = {
        "total_clips": len(all_results),
        "total_frames": sum(r["frame_count"] for r in all_results),
        "tesseract_available": TESSERACT_AVAILABLE,
        "clips": [{"name": r["clip"], "frames": r["frame_count"],
                    "modes": r["detected_modes"]} for r in all_results],
    }
    with open(OUTPUT_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {summary['total_frames']} frames processed across {summary['total_clips']} clips.")
    print(f"Outputs in: {OUTPUT_ROOT}")
    print(f"  calibration/  — UI region overlays ({len(all_results)} images)")
    print(f"  ocr/          — per-clip OCR + detection JSONs")
    print(f"  summary.json  — aggregate stats")


if __name__ == "__main__":
    main()
