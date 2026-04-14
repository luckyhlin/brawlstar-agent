#!/usr/bin/env python3
"""Auto-label frames and generate an interactive HTML review page.

1. Reads frames listed in review_manifest.json
2. Auto-classifies each as gameplay/menu/intro/bad using image heuristics
3. Generates an interactive HTML page where you click thumbnails to correct labels
4. The HTML page saves the final labels back to review_manifest.json via a download

Usage:
  uv run python scripts/auto-label-and-review.py capture/frames/<clip_name>
  uv run python scripts/auto-label-and-review.py --all   # process all clips
"""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")


def classify_frame(frame: np.ndarray) -> tuple[str, float]:
    """Auto-classify a frame using image heuristics.

    Returns (label, confidence) where label is one of:
    gameplay, menu, intro, bad
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sat_mean = hsv[:, :, 1].mean()
    val_mean = hsv[:, :, 2].mean()
    val_std = gray.astype(float).std()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0

    # Check for large uniform regions (loading screens, solid backgrounds)
    uniform_ratio = (gray.astype(float).std() < 15)

    # Check for typical gameplay indicators:
    # - High saturation (colorful game map)
    # - High edge density (lots of objects, UI elements)
    # - Moderate-to-high brightness variance (not a solid screen)
    #
    # Menu/intro screens tend to have:
    # - Lower edge density OR very high (text-heavy)
    # - Sometimes lower saturation
    # - Large uniform color blocks

    score = 0.0

    # Saturation: gameplay maps are colorful
    if sat_mean > 60:
        score += 0.3
    elif sat_mean > 40:
        score += 0.15

    # Edge density: gameplay has moderate edges from terrain, characters
    if 0.05 < edge_density < 0.25:
        score += 0.3
    elif edge_density >= 0.25:
        score += 0.1  # might be text-heavy intro

    # Brightness variance: gameplay has varied lighting
    if val_std > 40:
        score += 0.2

    # Check corners for joystick indicators (gameplay has circular controls)
    bottom_left = frame[int(h*0.7):, :int(w*0.25)]
    bottom_right = frame[int(h*0.7):, int(w*0.75):]
    bl_sat = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2HSV)[:,:,1].mean()
    br_sat = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2HSV)[:,:,1].mean()
    if bl_sat > 30 and br_sat > 30:
        score += 0.2

    # Very dark frames are usually loading/transition
    if val_mean < 30:
        return ("bad", 0.8)

    if uniform_ratio:
        return ("intro", 0.6)

    if score >= 0.6:
        return ("gameplay", min(score, 0.95))
    elif score >= 0.35:
        return ("menu", 0.5)
    else:
        return ("intro", 0.5)


def frame_to_thumb_base64(frame: np.ndarray, max_w: int = 240) -> str:
    """Convert a frame to a base64-encoded JPEG thumbnail."""
    h, w = frame.shape[:2]
    scale = max_w / w
    thumb = cv2.resize(frame, (max_w, int(h * scale)))
    img = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def generate_html(clip_name: str, frames_dir: Path, manifest: dict) -> str:
    """Generate an interactive HTML labeling page."""
    items = []
    for fname, label in manifest["frame_labels"].items():
        fpath = frames_dir / fname
        if not fpath.exists():
            continue
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        # Apply crop if defined
        r = manifest.get("crop_region")
        if r and r.get("w") and r.get("h"):
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            frame = frame[y:y+h, x:x+w]

        auto_label, confidence = classify_frame(frame)
        if label == "unknown":
            label = auto_label

        thumb_b64 = frame_to_thumb_base64(frame)
        items.append({
            "name": fname,
            "label": label,
            "auto_label": auto_label,
            "confidence": round(confidence, 2),
            "thumb": thumb_b64,
        })

    manifest_path = str(frames_dir / "review_manifest.json")

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Review: {clip_name}</title>
<style>
  body {{ font-family: system-ui; background: #1a1a2e; color: #eee; margin: 20px; }}
  h1 {{ color: #e94560; }}
  .stats {{ margin: 10px 0; padding: 10px; background: #16213e; border-radius: 8px; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }}
  .card {{
    border: 3px solid #333; border-radius: 8px; padding: 5px;
    cursor: pointer; text-align: center; transition: all 0.15s;
    position: relative;
  }}
  .card:hover {{ transform: scale(1.05); }}
  .card img {{ display: block; border-radius: 4px; }}
  .card .fname {{ font-size: 11px; color: #888; margin-top: 3px; }}
  .card .label {{ font-size: 13px; font-weight: bold; margin-top: 2px; }}
  .card .auto {{ font-size: 10px; color: #666; }}
  .card[data-label="gameplay"] {{ border-color: #00d68f; }}
  .card[data-label="menu"] {{ border-color: #ffaa00; }}
  .card[data-label="intro"] {{ border-color: #6c757d; }}
  .card[data-label="bad"] {{ border-color: #e94560; }}
  .label-gameplay {{ color: #00d68f; }}
  .label-menu {{ color: #ffaa00; }}
  .label-intro {{ color: #6c757d; }}
  .label-bad {{ color: #e94560; }}
  .actions {{ margin: 15px 0; }}
  .actions button {{
    padding: 10px 20px; margin-right: 10px; border: none;
    border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold;
  }}
  .btn-save {{ background: #00d68f; color: #000; }}
  .btn-all-gameplay {{ background: #16213e; color: #00d68f; border: 2px solid #00d68f !important; }}
  .btn-all-bad {{ background: #16213e; color: #e94560; border: 2px solid #e94560 !important; }}
  .legend {{ display: flex; gap: 15px; margin: 10px 0; font-size: 13px; }}
  .legend span {{ padding: 3px 8px; border-radius: 4px; }}
</style>
</head><body>
<h1>Review: {clip_name}</h1>
<div class="legend">
  Click a frame to cycle: 
  <span style="background:#00d68f;color:#000">gameplay</span>
  <span style="background:#ffaa00;color:#000">menu</span>
  <span style="background:#6c757d;color:#fff">intro</span>
  <span style="background:#e94560;color:#fff">bad</span>
</div>
<div class="stats" id="stats"></div>
<div class="actions">
  <button class="btn-save" onclick="saveManifest()">Save review_manifest.json</button>
  <button class="btn-all-gameplay" onclick="setAll('gameplay')">All → gameplay</button>
  <button class="btn-all-bad" onclick="setAll('bad')">All → bad</button>
</div>
<div class="grid" id="grid"></div>

<script>
const LABELS = ["gameplay", "menu", "intro", "bad"];
const MANIFEST_PATH = {json.dumps(manifest_path)};
const CROP_REGION = {json.dumps(manifest.get("crop_region"))};
const items = {json.dumps(items)};

function renderGrid() {{
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  items.forEach((item, idx) => {{
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.label = item.label;
    card.onclick = () => {{
      const li = LABELS.indexOf(item.label);
      item.label = LABELS[(li + 1) % LABELS.length];
      card.dataset.label = item.label;
      card.querySelector(".label").className = "label label-" + item.label;
      card.querySelector(".label").textContent = item.label;
      updateStats();
    }};
    card.innerHTML = `
      <img src="data:image/jpeg;base64,${{item.thumb}}" />
      <div class="fname">${{item.name}}</div>
      <div class="label label-${{item.label}}">${{item.label}}</div>
      <div class="auto">auto: ${{item.auto_label}} (${{item.confidence}})</div>
    `;
    grid.appendChild(card);
  }});
  updateStats();
}}

function updateStats() {{
  const counts = {{}};
  items.forEach(i => {{ counts[i.label] = (counts[i.label] || 0) + 1; }});
  const parts = LABELS.map(l => `<span class="label-${{l}}">${{l}}: ${{counts[l] || 0}}</span>`);
  document.getElementById("stats").innerHTML = `${{items.length}} frames: ${{parts.join(" | ")}}`;
}}

function setAll(label) {{
  items.forEach(i => i.label = label);
  renderGrid();
}}

function saveManifest() {{
  const manifest = {{
    frames_dir: {json.dumps(manifest.get("frames_dir", ""))},
    crop_region: CROP_REGION,
    frame_labels: {{}}
  }};
  items.forEach(i => {{ manifest.frame_labels[i.name] = i.label; }});
  const blob = new Blob([JSON.stringify(manifest, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "review_manifest.json";
  a.click();
  alert("Saved! Move the downloaded file to:\\n" + MANIFEST_PATH);
}}

renderGrid();
</script>
</body></html>"""
    return html


def process_clip(frames_dir: Path):
    """Process a single clip directory."""
    manifest_path = frames_dir / "review_manifest.json"
    if not manifest_path.exists():
        print(f"  SKIP (no manifest): {frames_dir.name}")
        return

    manifest = json.load(open(manifest_path))
    clip_name = frames_dir.name

    print(f"  Processing: {clip_name} ({len(manifest['frame_labels'])} frames)")

    html = generate_html(clip_name, frames_dir, manifest)
    html_path = frames_dir / "review.html"
    html_path.write_text(html)

    # Also update the manifest with auto-labels for unknown frames
    updated = False
    for fname in manifest["frame_labels"]:
        if manifest["frame_labels"][fname] == "unknown":
            fpath = frames_dir / fname
            if fpath.exists():
                frame = cv2.imread(str(fpath))
                if frame is not None:
                    r = manifest.get("crop_region")
                    if r and r.get("w") and r.get("h"):
                        frame = frame[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
                    label, _ = classify_frame(frame)
                    manifest["frame_labels"][fname] = label
                    updated = True

    if updated:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"    → {html_path}")
    counts = {}
    for v in manifest["frame_labels"].values():
        counts[v] = counts.get(v, 0) + 1
    print(f"    Auto-labels: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", nargs="?", type=Path, help="Clip frames directory")
    parser.add_argument("--all", action="store_true", help="Process all clips")
    args = parser.parse_args()

    if args.all:
        frames_root = PROJECT_ROOT / "capture" / "frames"
        dirs = sorted(d for d in frames_root.iterdir()
                      if d.is_dir() and (d / "review_manifest.json").exists())
        print(f"Processing {len(dirs)} clips...")
        for d in dirs:
            process_clip(d)
    elif args.frames_dir:
        process_clip(args.frames_dir)
    else:
        print("Usage: auto-label-and-review.py <frames_dir> | --all")
        sys.exit(1)

    print("\nDone! Open the review.html files in your browser to review and correct labels.")
    print("After correcting, click 'Save' in the HTML page, then move the downloaded")
    print("review_manifest.json back to the clip's frames directory.")


if __name__ == "__main__":
    main()
