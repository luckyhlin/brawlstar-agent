"""Character matching — match in-game brawler appearances against portrait catalog.

This is a baseline using color histograms and template matching.
The 2.5D in-game view differs significantly from portrait references,
so this provides a starting point — not production accuracy.
"""

import json
from pathlib import Path
import cv2
import numpy as np

PORTRAITS_DIR = Path("/media/lin/disk2/brawlstar-agent/datasets/character_refs/portraits")
INDEX_FILE = PORTRAITS_DIR.parent / "brawlers_index.json"

_portrait_cache: dict[str, np.ndarray] = {}
_index_cache: list[dict] | None = None


def load_index() -> list[dict]:
    global _index_cache
    if _index_cache is None:
        if INDEX_FILE.exists():
            _index_cache = json.loads(INDEX_FILE.read_text())
        else:
            _index_cache = []
    return _index_cache


def load_portrait(brawler_id: int, variant: str = "borderless") -> np.ndarray | None:
    """Load a brawler portrait. Returns BGR image or None."""
    key = f"{brawler_id}_{variant}"
    if key not in _portrait_cache:
        path = PORTRAITS_DIR / f"{brawler_id}_{variant}.png"
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        _portrait_cache[key] = img
    return _portrait_cache[key]


def compute_color_histogram(img: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute a normalized HSV color histogram."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def find_similar_brawlers(
    crop: np.ndarray,
    top_k: int = 5,
    variant: str = "borderless",
) -> list[dict]:
    """Find the top-k most similar brawler portraits to a game crop.

    Uses color histogram comparison (Bhattacharyya distance).
    Returns list of {id, name, distance, rank}.
    """
    index = load_index()
    if not index:
        return []

    crop_hist = compute_color_histogram(crop)
    results = []

    for entry in index:
        bid = entry["id"]
        portrait = load_portrait(bid, variant)
        if portrait is None:
            continue
        port_hist = compute_color_histogram(portrait)
        dist = cv2.compareHist(
            crop_hist.reshape(-1, 1).astype(np.float32),
            port_hist.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        results.append({"id": bid, "name": entry.get("name", "?"), "distance": float(dist)})

    results.sort(key=lambda x: x["distance"])
    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1
    return results[:top_k]


def detect_brawler_blobs(frame: np.ndarray, min_area: int = 400) -> list[tuple[int, int, int, int]]:
    """Detect potential brawler locations as colored blobs in the game area.

    Returns list of (x, y, w, h) bounding boxes.
    Brawlers typically have high saturation and are surrounded by a colored ring.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Brawler selection rings are highly saturated
    sat_mask = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([180, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if 0.4 < aspect < 2.5:
            bboxes.append((x, y, w, h))
    return bboxes
