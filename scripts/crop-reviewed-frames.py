#!/usr/bin/env python3
"""Crop review-approved gameplay frames into a dataset directory.

Usage:
  uv run python scripts/crop-reviewed-frames.py capture/frames/sample_gameplay
  uv run python scripts/crop-reviewed-frames.py capture/frames/sample_gameplay --label gameplay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from brawlstar_agent.crop import crop_game_region


PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop gameplay frames from a review manifest")
    parser.add_argument("frames_dir", type=Path, help="Directory of extracted frames")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to review manifest JSON (defaults to <frames_dir>/review_manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for cropped frames (defaults to datasets/gameplay_cropped/<clip_name>)",
    )
    parser.add_argument(
        "--label",
        default="gameplay",
        help="Frame label to export from the manifest (default: gameplay)",
    )
    parser.add_argument(
        "--target-size",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        type=int,
        default=None,
        help="Optional resize for cropped frames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    manifest_path = (args.manifest or (frames_dir / "review_manifest.json")).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (PROJECT_ROOT / "datasets" / "gameplay_cropped" / frames_dir.name)
    )

    if not frames_dir.is_dir():
        print(f"Frames directory not found: {frames_dir}", file=sys.stderr)
        return 1

    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    crop_region = manifest.get("crop_region")
    if not crop_region:
        print(f"Manifest has no crop_region: {manifest_path}", file=sys.stderr)
        return 1

    region = (
        int(crop_region["x"]),
        int(crop_region["y"]),
        int(crop_region["w"]),
        int(crop_region["h"]),
    )
    target_size = tuple(args.target_size) if args.target_size else None
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    missing = 0
    for frame_name, label in sorted(manifest.get("frame_labels", {}).items()):
        if label != args.label:
            continue

        frame_path = frames_dir / frame_name
        if not frame_path.is_file():
            missing += 1
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            missing += 1
            continue

        cropped = crop_game_region(frame, region, target_size)
        cv2.imwrite(str(output_dir / frame_name), cropped)
        exported += 1

    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")
    print(f"Exported {exported} '{args.label}' frames")
    if missing:
        print(f"Skipped {missing} unreadable or missing frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
