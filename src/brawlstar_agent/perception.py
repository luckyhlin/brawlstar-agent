"""Basic perception utilities for Brawl Stars frames.

Provides region-of-interest cropping, color analysis, and template matching
as building blocks for more advanced game-state extraction.
"""

from pathlib import Path
import cv2
import numpy as np


# Approximate UI regions for 1080x1920 landscape Brawl Stars gameplay.
# These will need calibration against actual captured frames.
REGIONS_1080P_LANDSCAPE = {
    "minimap":        (850, 500, 1070, 710),   # (x1, y1, x2, y2)
    "health_bar":     (100, 50, 500, 80),
    "super_button":   (900, 350, 1000, 450),
    "joystick_left":  (50, 400, 250, 600),
    "joystick_right": (850, 400, 1050, 600),
    "timer":          (490, 10, 590, 40),
    "scoreboard":     (420, 10, 660, 50),
}


def crop_region(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a rectangular region from a frame. bbox = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def dominant_colors(
    frame: np.ndarray, k: int = 3, max_pixels: int = 10000
) -> np.ndarray:
    """Find the k dominant colors in a frame region using k-means.

    Returns an array of shape (k, 3) with BGR colors sorted by frequency.
    """
    pixels = frame.reshape(-1, 3).astype(np.float32)
    if len(pixels) > max_pixels:
        indices = np.random.choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    order = counts.argsort()[::-1]
    return centers[order].astype(np.uint8)


def color_mask(
    frame: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> np.ndarray:
    """Create a binary mask for pixels within an HSV color range."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))


def match_template(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Find all locations where template matches above threshold.

    Returns list of (x, y, confidence) tuples.
    """
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray, template_gray = frame, template

    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    matches = []
    for pt in zip(*locations[::-1]):
        matches.append((pt[0], pt[1], float(result[pt[1], pt[0]])))
    return matches


def detect_text_regions(frame: np.ndarray, min_area: int = 100) -> list[tuple[int, int, int, int]]:
    """Detect potential text regions using MSER.

    Returns list of bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    bboxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        if w * h >= min_area and 0.1 < w / max(h, 1) < 10:
            bboxes.append((x, y, w, h))
    return bboxes
