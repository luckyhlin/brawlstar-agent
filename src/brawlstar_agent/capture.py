"""Frame capture and extraction utilities."""

from pathlib import Path
import subprocess
import cv2
import numpy as np


PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")
CLIPS_DIR = PROJECT_ROOT / "capture" / "clips"
FRAMES_DIR = PROJECT_ROOT / "capture" / "frames"
SCREENSHOTS_DIR = PROJECT_ROOT / "capture" / "screenshots"


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    fps: float = 2.0,
) -> Path:
    """Extract frames from a video file using ffmpeg.

    Returns the output directory path.
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = FRAMES_DIR / video_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(output_dir / "frame_%06d.jpg"),
            "-hide_banner", "-loglevel", "warning",
        ],
        check=True,
    )
    return output_dir


def load_frame(path: str | Path) -> np.ndarray:
    """Load a single frame as a BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_frame_rgb(path: str | Path) -> np.ndarray:
    """Load a single frame as an RGB numpy array (for matplotlib/PIL)."""
    return cv2.cvtColor(load_frame(path), cv2.COLOR_BGR2RGB)


def iter_frames(directory: str | Path, pattern: str = "*.jpg"):
    """Iterate over frame files in sorted order, yielding (path, image) tuples."""
    directory = Path(directory)
    for p in sorted(directory.glob(pattern)):
        yield p, load_frame(p)


def video_frames(video_path: str | Path):
    """Iterate directly over frames of a video file without writing to disk.

    Yields (frame_index, image_bgr) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def video_info(video_path: str | Path) -> dict:
    """Get basic info about a video file."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
