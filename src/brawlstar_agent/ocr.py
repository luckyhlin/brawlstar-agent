"""OCR extraction for Brawl Stars UI text.

Extracts text from timer, scores, player names, and other UI elements.
Uses pytesseract when available, gracefully degrades otherwise.
"""

import shutil
import cv2
import numpy as np

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

if TESSERACT_AVAILABLE:
    import pytesseract


def preprocess_for_ocr(crop: np.ndarray, invert: bool = True) -> np.ndarray:
    """Preprocess a UI region crop for OCR.

    Game text is typically white/bright on dark background with outlines.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    # Upscale small regions for better OCR
    h, w = gray.shape[:2]
    if w < 200:
        scale = 200 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    if invert:
        thresh = 255 - thresh

    # Light morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def extract_text(crop: np.ndarray, config: str = "--psm 7 -c tessedit_char_whitelist=0123456789:% ") -> dict:
    """Extract text from a UI region crop.

    Returns dict with 'text', 'confidence', 'raw', 'available'.
    """
    if not TESSERACT_AVAILABLE:
        return {"text": "", "confidence": 0.0, "raw": "", "available": False}

    processed = preprocess_for_ocr(crop, invert=False)
    try:
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        texts = []
        confs = []
        for i, conf in enumerate(data["conf"]):
            c = int(conf)
            if c > 20 and data["text"][i].strip():
                texts.append(data["text"][i].strip())
                confs.append(c)
        text = " ".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return {"text": text, "confidence": avg_conf, "raw": str(data["text"]), "available": True}
    except Exception as e:
        return {"text": "", "confidence": 0.0, "raw": str(e), "available": True}


def extract_timer(timer_crop: np.ndarray) -> dict:
    """Extract timer text (e.g., '2:01', '0:45')."""
    return extract_text(timer_crop, config="--psm 7 -c tessedit_char_whitelist=0123456789: ")


def extract_number(crop: np.ndarray) -> dict:
    """Extract a pure number (scores, gem counts, HP percentages)."""
    return extract_text(crop, config="--psm 7 -c tessedit_char_whitelist=0123456789% ")


def extract_player_name(crop: np.ndarray) -> dict:
    """Extract player name text (alphanumeric + some special chars)."""
    return extract_text(crop, config="--psm 7")
