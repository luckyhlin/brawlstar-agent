"""Auto-detect and crop the game region from gameplay videos with overlays.

Many YouTube videos have facecam, borders, sponsor overlays etc.
This module detects the actual game area and crops it out.
"""

from pathlib import Path
import cv2
import numpy as np


def detect_game_region(
    frame: np.ndarray,
    min_area_ratio: float = 0.3,
) -> tuple[int, int, int, int] | None:
    """Detect the main game area in a frame with overlays.

    Looks for the largest bright, high-saturation rectangular region,
    which is typically the game viewport against darker overlay borders.

    Returns (x, y, w, h) or None if detection fails.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection to find region boundaries
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    min_area = h * w * min_area_ratio

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / max(ch, 1)
        if area > min_area and area > best_area and 1.2 < aspect < 2.5:
            best = (x, y, cw, ch)
            best_area = area

    return best


def detect_game_region_by_variance(
    frame: np.ndarray,
    sample_frames: list[np.ndarray] | None = None,
) -> tuple[int, int, int, int]:
    """Detect game region by finding the area with highest pixel variance.

    Static overlays (facecam border, sponsor logos) have low variance across
    frames, while the game area changes constantly.

    If sample_frames is provided, uses cross-frame variance for better accuracy.
    Returns (x, y, w, h).
    """
    if sample_frames and len(sample_frames) >= 2:
        stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                          for f in sample_frames])
        variance_map = stack.var(axis=0)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        local_mean = cv2.blur(gray, (64, 64))
        variance_map = (gray - local_mean) ** 2

    # Threshold to find high-variance region
    norm = (variance_map / max(variance_map.max(), 1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(norm, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = frame.shape[:2]
        return (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def crop_game_region(
    frame: np.ndarray,
    region: tuple[int, int, int, int],
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Crop and optionally resize the game region.

    region: (x, y, w, h)
    target_size: (width, height) or None to keep original size
    """
    x, y, w, h = region
    cropped = frame[y:y+h, x:x+w]
    if target_size is not None:
        cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return cropped


def interactive_crop(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Let the user draw a rectangle on the frame to define the game region.

    Returns (x, y, w, h). For use in scripts, not notebooks.
    """
    clone = frame.copy()
    roi = cv2.selectROI("Select Game Region (press ENTER to confirm)", clone, fromCenter=False)
    cv2.destroyAllWindows()
    return roi  # (x, y, w, h)


def batch_crop_frames(
    input_dir: Path,
    output_dir: Path,
    region: tuple[int, int, int, int],
    pattern: str = "*.jpg",
    target_size: tuple[int, int] | None = None,
) -> int:
    """Crop all frames in a directory to the specified region.

    Returns the number of frames processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(Path(input_dir).glob(pattern)):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        cropped = crop_game_region(frame, region, target_size)
        cv2.imwrite(str(output_dir / p.name), cropped)
        count += 1
    return count
