#!/usr/bin/env python3
"""Prepare a frame directory for headless human review.

Creates:
- review_manifest.json with an auto-detected crop region
- review_sheet.jpg showing evenly sampled frames
- review_sheet_cropped.jpg showing the sampled frames after crop
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from brawlstar_agent.crop import crop_game_region, detect_game_region_by_variance


PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")


def sample_frame_paths(frames_dir: Path, sample_n: int) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return []
    if sample_n >= len(frames):
        return frames
    indices = np.linspace(0, len(frames) - 1, sample_n, dtype=int)
    return [frames[i] for i in indices]


def render_contact_sheet(
    frame_paths: list[Path],
    output_path: Path,
    crop_region: tuple[int, int, int, int] | None = None,
    thumb_size: tuple[int, int] = (320, 180),
    columns: int = 4,
) -> None:
    if not frame_paths:
        return

    tiles: list[np.ndarray] = []
    for idx, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if crop_region is not None:
            frame = crop_game_region(frame, crop_region)

        tile = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)
        label = f"{idx:02d} {frame_path.name}"
        cv2.rectangle(tile, (0, 0), (thumb_size[0], 26), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label[:40],
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return

    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    grid_rows: list[np.ndarray] = []
    for row_idx in range(rows):
        row_tiles = tiles[row_idx * columns:(row_idx + 1) * columns]
        if len(row_tiles) < columns:
            row_tiles.extend([blank.copy() for _ in range(columns - len(row_tiles))])
        grid_rows.append(cv2.hconcat(row_tiles))

    sheet = cv2.vconcat(grid_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare headless review assets for a frame directory")
    parser.add_argument("frames_dir", type=Path, help="Directory containing extracted frame JPGs")
    parser.add_argument("--sample", type=int, default=20, help="Number of evenly spaced frames to sample")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to review manifest JSON (defaults to <frames_dir>/review_manifest.json)",
    )
    args = parser.parse_args()

    frames_dir = args.frames_dir.resolve()
    manifest_path = (args.manifest or (frames_dir / "review_manifest.json")).resolve()
    frame_paths = sample_frame_paths(frames_dir, args.sample)
    if not frame_paths:
        raise SystemExit(f"No JPG frames found in {frames_dir}")

    frames = [cv2.imread(str(path)) for path in frame_paths]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise SystemExit(f"Unable to read frames from {frames_dir}")

    manifest: dict[str, object] = {
        "frames_dir": str(frames_dir.relative_to(PROJECT_ROOT)),
        "crop_region": None,
        "frame_labels": {},
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    crop_region = manifest.get("crop_region")
    if crop_region:
        region = (
            int(crop_region["x"]),
            int(crop_region["y"]),
            int(crop_region["w"]),
            int(crop_region["h"]),
        )
    else:
        region = detect_game_region_by_variance(frames[0], sample_frames=frames)
        manifest["crop_region"] = {
            "x": int(region[0]),
            "y": int(region[1]),
            "w": int(region[2]),
            "h": int(region[3]),
        }

    frame_labels = dict(manifest.get("frame_labels", {}))
    for path in frame_paths:
        frame_labels.setdefault(path.name, "unknown")
    manifest["frame_labels"] = frame_labels

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    render_contact_sheet(frame_paths, frames_dir / "review_sheet.jpg")
    render_contact_sheet(frame_paths, frames_dir / "review_sheet_cropped.jpg", crop_region=region)

    print(f"Prepared review assets for: {frames_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Crop region: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
    print(f"Sample sheet: {frames_dir / 'review_sheet.jpg'}")
    print(f"Cropped sheet: {frames_dir / 'review_sheet_cropped.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
