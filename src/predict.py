"""
predict.py
Core inference logic used by both the API and standalone CLI.

Usage (CLI):
    python src/predict.py path/to/image.png
"""

import sys
import joblib
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    preprocess_pipeline,
    load_image,
    load_image_from_bytes,
)
from src.feature_extraction import extract_all_features, get_display_features

MODEL_PATH = Path("model/drawing_detector.joblib")
LABEL_NAMES = {0: "Human", 1: "AI"}
UNCERTAINTY_THRESHOLD = 0.30   # if max confidence < 30%, return "Uncertain"


@dataclass
class PredictionResult:
    label: str           # "Human" | "AI" | "Uncertain"
    confidence: float    # 0.0 – 1.0
    human_prob: float
    ai_prob: float
    display_features: dict
    raw_features: dict


def load_model():
    """Load the trained model pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


_model_cache = None  # module-level cache to avoid reloading on every request


def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache


def predict_from_bgr(img_bgr: np.ndarray) -> PredictionResult:
    """
    Run full inference on a BGR numpy array.

    Parameters
    ----------
    img_bgr : uint8 BGR image (any size)

    Returns
    -------
    PredictionResult dataclass
    """
    model = get_model()

    # Preprocessing
    gray, _ = preprocess_pipeline(img_bgr)
    color_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    # Feature extraction
    feat_vec, feat_dict = extract_all_features(gray, color_bgr)
    display_feats = get_display_features(feat_dict)

    # Inference
    feat_2d = feat_vec.reshape(1, -1)
    proba = model.predict_proba(feat_2d)[0]  # [human_prob, ai_prob]

    human_prob = float(proba[0])
    ai_prob = float(proba[1])
    max_conf = max(human_prob, ai_prob)

    if max_conf < UNCERTAINTY_THRESHOLD:
        label = "Uncertain"
        confidence = max_conf
    elif ai_prob > human_prob:
        label = "AI"
        confidence = ai_prob
    else:
        label = "Human"
        confidence = human_prob

    return PredictionResult(
        label=label,
        confidence=confidence,
        human_prob=human_prob,
        ai_prob=ai_prob,
        display_features=display_feats,
        raw_features=feat_dict,
    )


def predict_from_bytes(image_bytes: bytes) -> PredictionResult:
    """Run inference on raw image bytes (from HTTP upload)."""
    img = load_image_from_bytes(image_bytes)
    return predict_from_bgr(img)


def predict_from_path(image_path: str) -> PredictionResult:
    """Run inference on an image file path."""
    img = load_image(image_path)
    return predict_from_bgr(img)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _print_result(result: PredictionResult):
    bar_len = 30

    def bar(prob):
        filled = int(round(prob * bar_len))
        return "█" * filled + "░" * (bar_len - filled)

    print("\n┌" + "─" * 44 + "┐")
    print("│{:^44}│".format("AI vs Human Drawing Detector"))
    print("├" + "─" * 44 + "┤")
    print(f"│  Prediction : {result.label:<29}│")
    print(f"│  Confidence : {result.confidence * 100:>6.2f}%{'':<22}│")
    print("├" + "─" * 44 + "┤")
    print(f"│  Human  {bar(result.human_prob)}  {result.human_prob * 100:>5.1f}%  │")
    print(f"│  AI     {bar(result.ai_prob)}  {result.ai_prob * 100:>5.1f}%  │")
    print("├" + "─" * 44 + "┤")
    print("│  Image Analysis{:<28}│".format(""))
    for k, v in result.display_features.items():
        print(f"│    {k:<22} {v:<14.4f}│")
    print("└" + "─" * 44 + "┘\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        result = predict_from_path(image_path)
        _print_result(result)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
