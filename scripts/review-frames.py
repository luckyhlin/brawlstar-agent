#!/usr/bin/env python3
"""Human-in-the-loop frame review tool.

Displays frames from a directory in a grid, lets you:
- Mark the game crop region on the first frame
- Classify frames as gameplay / menu / intro / bad
- Save selections to a JSON manifest

Usage:
  uv run python scripts/review-frames.py capture/frames/sample_gameplay/
  uv run python scripts/review-frames.py capture/frames/sample_gameplay/ --crop
  uv run python scripts/review-frames.py capture/frames/sample_gameplay/ --sample 20
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")


def select_crop_region(frames_dir: Path) -> tuple[int, int, int, int] | None:
    """Show a frame and let the user draw the game region."""
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print("No frames found.")
        return None

    # Pick a frame from the middle (more likely to be gameplay)
    mid = len(frames) // 2
    frame = cv2.imread(str(frames[mid]))
    if frame is None:
        return None

    print("Draw a rectangle around the GAME AREA, then press ENTER.")
    print("Press C to cancel.")
    roi = cv2.selectROI("Select Game Region", frame, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        print("No region selected.")
        return None

    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    print(f"Selected region: x={x}, y={y}, w={w}, h={h}")
    return (x, y, w, h)


def classify_frames(
    frames_dir: Path,
    sample_n: int | None = None,
) -> dict:
    """Show frames one by one for classification.

    Keys:
      g = gameplay (good)
      m = menu / loading screen
      i = intro / outro
      b = bad / blurry / useless
      q = quit
      SPACE = skip (keep previous label or 'unknown')
    """
    frames = sorted(frames_dir.glob("*.jpg"))
    if sample_n and sample_n < len(frames):
        indices = np.linspace(0, len(frames) - 1, sample_n, dtype=int)
        frames = [frames[i] for i in indices]

    labels = {}
    label_map = {
        ord('g'): "gameplay",
        ord('m'): "menu",
        ord('i'): "intro",
        ord('b'): "bad",
        ord(' '): "unknown",
    }

    print(f"\nClassifying {len(frames)} frames.")
    print("Keys: g=gameplay, m=menu, i=intro, b=bad, SPACE=skip, q=quit")

    for idx, p in enumerate(frames):
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        display = cv2.resize(frame, (960, 540))
        cv2.putText(display, f"[{idx+1}/{len(frames)}] {p.name}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "g=gameplay m=menu i=intro b=bad SPACE=skip q=quit",
                     (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Frame Review", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        label = label_map.get(key, "unknown")
        labels[p.name] = label
        print(f"  {p.name} → {label}")

    cv2.destroyAllWindows()
    return labels


def main():
    parser = argparse.ArgumentParser(description="Review and annotate extracted frames")
    parser.add_argument("frames_dir", type=Path, help="Directory of extracted frames")
    parser.add_argument("--crop", action="store_true", help="Select crop region first")
    parser.add_argument("--sample", type=int, default=None, help="Review N evenly sampled frames")
    parser.add_argument("--output", type=Path, default=None, help="Output manifest JSON path")
    args = parser.parse_args()

    if not args.frames_dir.is_dir():
        print(f"Not a directory: {args.frames_dir}")
        sys.exit(1)

    manifest = {
        "frames_dir": str(args.frames_dir),
        "crop_region": None,
        "frame_labels": {},
    }

    if args.crop:
        region = select_crop_region(args.frames_dir)
        if region:
            manifest["crop_region"] = {"x": region[0], "y": region[1], "w": region[2], "h": region[3]}

    labels = classify_frames(args.frames_dir, args.sample)
    manifest["frame_labels"] = labels

    output_path = args.output or (args.frames_dir / "review_manifest.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {output_path}")

    gameplay_count = sum(1 for v in labels.values() if v == "gameplay")
    total = len(labels)
    print(f"Gameplay frames: {gameplay_count}/{total}")


if __name__ == "__main__":
    main()
