"""UI region definitions and calibration for Brawl Stars gameplay frames.

All coordinates are normalized [0.0, 1.0] so they work across resolutions.
The game has multiple modes with different UI layouts, but common elements exist.
"""

import cv2
import numpy as np

# Normalized regions common across most game modes.
# Format: (x_frac, y_frac, w_frac, h_frac) as fractions of frame width/height.
COMMON_REGIONS = {
    "top_bar":        (0.10, 0.00, 0.80, 0.07),
    "timer":          (0.42, 0.00, 0.16, 0.06),
    "top_left_info":  (0.00, 0.00, 0.25, 0.07),
    "top_right_info": (0.75, 0.00, 0.25, 0.07),
    "joystick_left":  (0.00, 0.60, 0.20, 0.40),
    "joystick_right": (0.75, 0.55, 0.25, 0.45),
    "super_button":   (0.80, 0.45, 0.12, 0.18),
    "gadget_button":  (0.88, 0.35, 0.08, 0.12),
    "game_area":      (0.10, 0.05, 0.70, 0.60),
    "bottom_bar":     (0.00, 0.85, 1.00, 0.15),
    "chat_bubble":    (0.90, 0.28, 0.08, 0.08),
}

# Mode-specific regions (extend as we identify more modes)
MODE_REGIONS = {
    "showdown": {
        "brawlers_left": (0.00, 0.00, 0.22, 0.06),
    },
    "gem_grab": {
        "gem_count":     (0.42, 0.00, 0.16, 0.06),
        "blue_gems":     (0.35, 0.00, 0.10, 0.05),
        "red_gems":      (0.55, 0.00, 0.10, 0.05),
    },
    "brawl_ball": {
        "blue_score":    (0.05, 0.85, 0.35, 0.15),
        "red_score":     (0.60, 0.85, 0.35, 0.15),
    },
    "heist": {
        "blue_safe_hp":  (0.05, 0.00, 0.20, 0.04),
        "red_safe_hp":   (0.75, 0.00, 0.20, 0.04),
    },
}


def norm_to_pixel(region_norm: tuple, w: int, h: int) -> tuple[int, int, int, int]:
    """Convert normalized (x_frac, y_frac, w_frac, h_frac) to pixel (x, y, w, h)."""
    xf, yf, wf, hf = region_norm
    return (int(xf * w), int(yf * h), int(wf * w), int(hf * h))


def crop_normalized(frame: np.ndarray, region_norm: tuple) -> np.ndarray:
    """Crop a frame using normalized coordinates."""
    h, w = frame.shape[:2]
    x, y, rw, rh = norm_to_pixel(region_norm, w, h)
    return frame[y:y+rh, x:x+rw]


def detect_game_mode(frame: np.ndarray) -> str:
    """Try to detect the game mode from UI cues.

    Uses color and text patterns in key regions.
    Returns one of: showdown, gem_grab, brawl_ball, heist, unknown
    """
    h, w = frame.shape[:2]

    # Check for "Brawlers left" text area (showdown)
    top_left = crop_normalized(frame, (0.0, 0.0, 0.25, 0.06))
    tl_gray = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
    white_pixels = (tl_gray > 200).sum() / tl_gray.size
    if white_pixels > 0.05:
        return "showdown"

    # Check for bottom score panels (brawl_ball)
    bottom = crop_normalized(frame, (0.0, 0.80, 1.0, 0.20))
    hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
    blue_ratio = blue_mask.sum() / (blue_mask.size * 255)
    if blue_ratio > 0.08:
        return "brawl_ball"

    # Check for gem icon in timer area (gem_grab)
    timer_area = crop_normalized(frame, (0.42, 0.0, 0.16, 0.06))
    timer_hsv = cv2.cvtColor(timer_area, cv2.COLOR_BGR2HSV)
    purple_mask = cv2.inRange(timer_hsv, np.array([120, 50, 50]), np.array([160, 255, 255]))
    if purple_mask.sum() / (purple_mask.size * 255) > 0.02:
        return "gem_grab"

    # Check for health bar color patterns (heist)
    top_right = crop_normalized(frame, (0.75, 0.0, 0.25, 0.05))
    tr_hsv = cv2.cvtColor(top_right, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(tr_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    if red_mask.sum() / (red_mask.size * 255) > 0.05:
        return "heist"

    return "unknown"


def draw_regions_overlay(
    frame: np.ndarray,
    regions: dict[str, tuple] | None = None,
    mode: str | None = None,
) -> np.ndarray:
    """Draw labeled rectangles for all UI regions on a frame copy.

    Returns annotated frame (does not modify original).
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    all_regions = dict(COMMON_REGIONS)
    if mode and mode in MODE_REGIONS:
        all_regions.update(MODE_REGIONS[mode])
    if regions:
        all_regions.update(regions)

    colors = {
        "timer": (0, 255, 255),
        "game_area": (0, 255, 0),
        "joystick_left": (255, 100, 100),
        "joystick_right": (255, 100, 100),
        "super_button": (255, 0, 255),
    }
    default_color = (0, 200, 255)

    for name, norm_rect in all_regions.items():
        x, y, rw, rh = norm_to_pixel(norm_rect, w, h)
        color = colors.get(name, default_color)
        cv2.rectangle(vis, (x, y), (x + rw, y + rh), color, 2)
        cv2.putText(vis, name, (x + 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return vis
