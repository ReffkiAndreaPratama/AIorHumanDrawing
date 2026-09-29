"""
preprocessing.py
Handles image loading, validation, and normalization
before feature extraction.
"""

import cv2
import numpy as np
from pathlib import Path


# Supported image extensions
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Target size for all images
TARGET_SIZE = (256, 256)


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk. Returns BGR numpy array.
    Raises ValueError if file is unsupported or unreadable.
    """
    path = Path(image_path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from raw bytes (e.g. from FastAPI UploadFile).
    Returns BGR numpy array.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from bytes.")
    return img


def resize_image(img: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
    """Resize image to a fixed square size using INTER_AREA for shrinking."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 2:
        return img  # already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] float32.
    Input can be grayscale or color.
    """
    return img.astype(np.float32) / 255.0


def preprocess_pipeline(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.

    Returns:
        gray  — resized grayscale uint8 image (for feature extraction)
        color — resized color float32 image normalized to [0,1]
    """
    resized = resize_image(img)
    gray = to_grayscale(resized)
    color = normalize_image(resized)
    return gray, color


def preprocess_from_bytes(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: bytes → preprocessed (gray, color)."""
    img = load_image_from_bytes(image_bytes)
    return preprocess_pipeline(img)
