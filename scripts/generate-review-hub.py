#!/usr/bin/env python3
"""Generate a single HTML review hub that handles all clips in-browser.

No terminal interaction needed — everything happens in the browser:
- Navigate between clips with Next/Previous
- Click thumbnails to label
- Save writes directly via a tiny local server running in the background

Usage:
  uv run python scripts/generate-review-hub.py
  # Then open http://localhost:8787 in your browser
"""

import http.server
import json
import threading
import webbrowser
import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs

import cv2
from PIL import Image

PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")
FRAMES_ROOT = PROJECT_ROOT / "capture" / "frames"
PORT = 8787


def classify_frame(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sat_mean = hsv[:, :, 1].mean()
    val_mean = hsv[:, :, 2].mean()
    val_std = gray.astype(float).std()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    score = 0.0
    if sat_mean > 60: score += 0.3
    elif sat_mean > 40: score += 0.15
    if 0.05 < edge_density < 0.25: score += 0.3
    elif edge_density >= 0.25: score += 0.1
    if val_std > 40: score += 0.2
    bl = frame[int(h*0.7):, :int(w*0.25)]
    br = frame[int(h*0.7):, int(w*0.75):]
    if cv2.cvtColor(bl, cv2.COLOR_BGR2HSV)[:,:,1].mean() > 30 and \
       cv2.cvtColor(br, cv2.COLOR_BGR2HSV)[:,:,1].mean() > 30:
        score += 0.2
    if val_mean < 30: return "bad", 0.8
    if score >= 0.6: return "gameplay", min(score, 0.95)
    elif score >= 0.35: return "menu", 0.5
    return "intro", 0.5


def frame_to_thumb(frame, max_w=220):
    h, w = frame.shape[:2]
    thumb = cv2.resize(frame, (max_w, int(h * max_w / w)))
    img = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def build_clips_data():
    clips = []
    dirs = sorted(d for d in FRAMES_ROOT.iterdir()
                  if d.is_dir() and (d / "review_manifest.json").exists())
    for d in dirs:
        manifest = json.loads((d / "review_manifest.json").read_text())
        frames = []
        for fname, label in manifest.get("frame_labels", {}).items():
            fpath = d / fname
            if not fpath.exists():
                continue
            frame = cv2.imread(str(fpath))
            if frame is None:
                continue
            r = manifest.get("crop_region")
            if r and r.get("w") and r.get("h"):
                frame = frame[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
            auto_label, conf = classify_frame(frame)
            if label == "unknown":
                label = auto_label
            frames.append({
                "name": fname,
                "label": label,
                "auto_label": auto_label,
                "confidence": round(conf, 2),
                "thumb": frame_to_thumb(frame),
            })
        clips.append({
            "dir_name": d.name,
            "dir_path": str(d),
            "manifest_path": str(d / "review_manifest.json"),
            "crop_region": manifest.get("crop_region"),
            "frames_dir": manifest.get("frames_dir", ""),
            "frames": frames,
        })
    return clips


def generate_hub_html(clips_json: str) -> str:
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Brawl Stars Review Hub</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui; background: #0f0f23; color: #eee; margin: 0; padding: 15px; }}
  h1 {{ color: #e94560; margin: 0 0 10px; font-size: 22px; }}
  .topbar {{
    position: sticky; top: 0; z-index: 100; background: #0f0f23;
    padding: 10px 0; border-bottom: 1px solid #333;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }}
  .topbar button {{
    padding: 8px 16px; border: none; border-radius: 6px;
    cursor: pointer; font-size: 13px; font-weight: bold;
  }}
  .btn-nav {{ background: #16213e; color: #eee; }}
  .btn-nav:hover {{ background: #1a2744; }}
  .btn-save {{ background: #00d68f; color: #000; }}
  .btn-save:hover {{ background: #00c07b; }}
  .btn-save.saved {{ background: #555; color: #aaa; }}
  .btn-action {{ background: #16213e; color: #ffaa00; border: 1px solid #ffaa00 !important; }}
  .clip-title {{ font-size: 15px; color: #ccc; flex: 1; min-width: 200px; }}
  .clip-title .idx {{ color: #e94560; }}
  .stats {{ font-size: 13px; color: #888; }}
  .sidebar {{
    position: fixed; right: 0; top: 0; width: 220px; height: 100vh;
    background: #16213e; overflow-y: auto; padding: 10px; z-index: 200;
    border-left: 1px solid #333; font-size: 12px;
  }}
  .sidebar h3 {{ color: #e94560; margin: 0 0 8px; font-size: 14px; }}
  .sidebar .clip-link {{
    display: block; padding: 5px 8px; margin: 2px 0; border-radius: 4px;
    cursor: pointer; color: #ccc; text-decoration: none;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .sidebar .clip-link:hover {{ background: #1a2744; }}
  .sidebar .clip-link.active {{ background: #e94560; color: #fff; }}
  .sidebar .clip-link .badge {{
    display: inline-block; font-size: 10px; padding: 1px 5px;
    border-radius: 3px; margin-left: 4px;
  }}
  .badge-done {{ background: #00d68f; color: #000; }}
  .badge-partial {{ background: #ffaa00; color: #000; }}
  .main {{ margin-right: 230px; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
  .card {{
    border: 3px solid #333; border-radius: 6px; padding: 4px;
    cursor: pointer; text-align: center; transition: all 0.12s;
  }}
  .card:hover {{ transform: scale(1.05); }}
  .card img {{ display: block; border-radius: 3px; }}
  .card .fname {{ font-size: 10px; color: #666; margin-top: 2px; }}
  .card .lbl {{ font-size: 12px; font-weight: bold; margin-top: 1px; }}
  .card[data-label="gameplay"] {{ border-color: #00d68f; }}
  .card[data-label="menu"] {{ border-color: #ffaa00; }}
  .card[data-label="intro"] {{ border-color: #6c757d; }}
  .card[data-label="bad"] {{ border-color: #e94560; }}
  .l-gameplay {{ color: #00d68f; }}
  .l-menu {{ color: #ffaa00; }}
  .l-intro {{ color: #6c757d; }}
  .l-bad {{ color: #e94560; }}
  .legend {{ font-size: 12px; color: #888; margin-top: 5px; }}
</style>
</head><body>

<div class="sidebar" id="sidebar"></div>
<div class="main">
  <div class="topbar" id="topbar"></div>
  <div class="legend">Click frame to cycle: gameplay → menu → intro → bad</div>
  <div class="grid" id="grid"></div>
</div>

<script>
const LABELS = ["gameplay", "menu", "intro", "bad"];
const clips = {clips_json};
let currentIdx = 0;
let dirty = new Set();

function renderSidebar() {{
  const sb = document.getElementById("sidebar");
  sb.innerHTML = "<h3>Clips (" + clips.length + ")</h3>";
  clips.forEach((clip, idx) => {{
    const gp = clip.frames.filter(f => f.label === "gameplay").length;
    const total = clip.frames.length;
    const unk = clip.frames.filter(f => f.label === "unknown").length;
    const badge = unk > 0 ? '<span class="badge badge-partial">?</span>' :
                  '<span class="badge badge-done">✓</span>';
    const a = document.createElement("a");
    a.className = "clip-link" + (idx === currentIdx ? " active" : "");
    a.innerHTML = (idx+1) + ". " + clip.dir_name.substring(0, 35) + badge;
    a.onclick = () => {{ switchClip(idx); }};
    sb.appendChild(a);
  }});
}}

function renderTopbar() {{
  const clip = clips[currentIdx];
  const gp = clip.frames.filter(f => f.label === "gameplay").length;
  const tb = document.getElementById("topbar");
  tb.innerHTML = `
    <button class="btn-nav" onclick="switchClip(currentIdx-1)" ${{currentIdx===0?'disabled':''}}>← Prev</button>
    <button class="btn-nav" onclick="switchClip(currentIdx+1)" ${{currentIdx===clips.length-1?'disabled':''}}>Next →</button>
    <button class="btn-save ${{dirty.has(currentIdx)?'':'saved'}}" onclick="saveCurrentClip()">
      ${{dirty.has(currentIdx)?'Save':'Saved ✓'}}
    </button>
    <button class="btn-action" onclick="setAllCurrent('gameplay')">All→gameplay</button>
    <button class="btn-action" onclick="setAllCurrent('bad')">All→bad</button>
    <span class="clip-title"><span class="idx">[${{currentIdx+1}}/${{clips.length}}]</span> ${{clip.dir_name}}</span>
    <span class="stats">gameplay: ${{gp}}/${{clip.frames.length}}</span>
  `;
}}

function renderGrid() {{
  const clip = clips[currentIdx];
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  clip.frames.forEach((item, fi) => {{
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.label = item.label;
    card.onclick = () => {{
      const li = LABELS.indexOf(item.label);
      item.label = LABELS[(li + 1) % LABELS.length];
      card.dataset.label = item.label;
      card.querySelector(".lbl").className = "lbl l-" + item.label;
      card.querySelector(".lbl").textContent = item.label;
      dirty.add(currentIdx);
      renderTopbar();
    }};
    card.innerHTML = `<img src="data:image/jpeg;base64,${{item.thumb}}" />
      <div class="fname">${{item.name}}</div>
      <div class="lbl l-${{item.label}}">${{item.label}}</div>`;
    grid.appendChild(card);
  }});
}}

function switchClip(idx) {{
  if (idx < 0 || idx >= clips.length) return;
  if (dirty.has(currentIdx)) {{
    if (!confirm("Unsaved changes on current clip. Switch anyway?")) return;
  }}
  currentIdx = idx;
  renderSidebar();
  renderTopbar();
  renderGrid();
  window.scrollTo(0, 0);
}}

function setAllCurrent(label) {{
  clips[currentIdx].frames.forEach(f => f.label = label);
  dirty.add(currentIdx);
  renderTopbar();
  renderGrid();
}}

function saveCurrentClip() {{
  const clip = clips[currentIdx];
  const manifest = {{
    frames_dir: clip.frames_dir,
    crop_region: clip.crop_region,
    frame_labels: {{}}
  }};
  clip.frames.forEach(f => {{ manifest.frame_labels[f.name] = f.label; }});
  const body = JSON.stringify({{
    path: clip.manifest_path,
    content: manifest
  }});
  fetch("/save", {{ method: "POST", body: body, headers: {{"Content-Type": "application/json"}} }})
    .then(r => r.json())
    .then(data => {{
      if (data.ok) {{
        dirty.delete(currentIdx);
        renderTopbar();
        renderSidebar();
      }} else {{
        alert("Save failed: " + (data.error || "unknown"));
      }}
    }})
    .catch(e => alert("Save error: " + e));
}}

// Keyboard shortcuts
document.addEventListener("keydown", e => {{
  if (e.key === "ArrowRight") switchClip(currentIdx + 1);
  else if (e.key === "ArrowLeft") switchClip(currentIdx - 1);
  else if (e.key === "s" && (e.ctrlKey || e.metaKey)) {{ e.preventDefault(); saveCurrentClip(); }}
}});

renderSidebar();
renderTopbar();
renderGrid();
</script>
</body></html>"""


class ReviewHandler(http.server.BaseHTTPRequestHandler):
    hub_html = ""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.hub_html.encode())

    def do_POST(self):
        if self.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            path = Path(body["path"])
            try:
                with open(path, "w") as f:
                    json.dump(body["content"], f, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode())
                print(f"  Saved: {path.parent.name}")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


def main():
    print("Building review hub (loading thumbnails for all clips)...")
    clips = build_clips_data()
    print(f"Loaded {len(clips)} clips, {sum(len(c['frames']) for c in clips)} total frames.")

    clips_json = json.dumps(clips)
    hub_html = generate_hub_html(clips_json)
    ReviewHandler.hub_html = hub_html

    server = http.server.HTTPServer(("127.0.0.1", PORT), ReviewHandler)
    print(f"\nReview hub running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")

    webbrowser.open(f"http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
