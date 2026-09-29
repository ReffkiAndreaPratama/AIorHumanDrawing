"""
train.py
Trains a classical ML classifier (Random Forest + SVM ensemble)
on image features extracted from the dataset.

Usage:
    python src/train.py

Dataset structure expected:
    dataset/
      train/
        human/   ← images of human drawings
        ai/      ← images of AI-generated drawings
      validation/
        human/
        ai/
"""

import os
import sys
import joblib
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_pipeline, load_image
from src.feature_extraction import extract_all_features

# ─── Config ───────────────────────────────────────────
DATASET_DIR = Path("dataset")
MODEL_OUTPUT = Path("model/drawing_detector.joblib")
SCALER_OUTPUT = Path("model/scaler.joblib")
LABEL_MAP = {"human": 0, "ai": 1}
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────


def load_dataset(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from dataset/<split>/{human,ai} and extract features.

    Returns
    -------
    X : float32 array of shape (n_samples, n_features)
    y : int array of shape (n_samples,)
    """
    X, y = [], []
    split_dir = DATASET_DIR / split

    for class_name, label in LABEL_MAP.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"  [WARN] Directory not found: {class_dir}")
            continue

        image_paths = list(class_dir.glob("*"))
        image_paths = [p for p in image_paths
                       if p.suffix.lower() in
                       {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

        print(f"  Loading {len(image_paths)} images from {class_dir} ...")

        for img_path in image_paths:
            try:
                img = load_image(str(img_path))
                gray, color_norm = preprocess_pipeline(img)
                # color_bgr needed for color features (uint8, not normalized)
                color_bgr = cv2.resize(img, (256, 256),
                                       interpolation=cv2.INTER_AREA)
                feat_vec, _ = extract_all_features(gray, color_bgr)
                X.append(feat_vec)
                y.append(label)
            except Exception as e:
                print(f"  [SKIP] {img_path.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model() -> Pipeline:
    """
    Build a Voting ensemble of SVM + Random Forest, wrapped in a
    StandardScaler pipeline.
    """
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    voting = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        voting="soft",
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", voting),
    ])
    return pipeline


def main():
    print("=" * 60)
    print(" AI vs Human Drawing Detector — Training")
    print("=" * 60)

    # ── Load training data ─────────────────────────────
    print("\n[1/4] Loading training data ...")
    X_train, y_train = load_dataset("train")

    if len(X_train) == 0:
        print("\n[ERROR] No training data found.")
        print("  Place images in dataset/train/human/ and dataset/train/ai/")
        sys.exit(1)

    print(f"  Loaded {len(X_train)} samples | "
          f"Human: {int(np.sum(y_train == 0))} | "
          f"AI: {int(np.sum(y_train == 1))}")

    # ── Load validation data ───────────────────────────
    print("\n[2/4] Loading validation data ...")
    X_val, y_val = load_dataset("validation")
    print(f"  Loaded {len(X_val)} validation samples")

    # ── Train ──────────────────────────────────────────
    print("\n[3/4] Training model ...")
    model = build_model()
    model.fit(X_train, y_train)
    print("  Done.")

    # ── Evaluate ───────────────────────────────────────
    print("\n[4/4] Evaluation ...")
    train_acc = model.score(X_train, y_train)
    print(f"  Train Accuracy : {train_acc * 100:.2f}%")

    if len(X_val) > 0:
        val_acc = model.score(X_val, y_val)
        print(f"  Val   Accuracy : {val_acc * 100:.2f}%")

        # Per-class report
        from sklearn.metrics import classification_report
        y_pred = model.predict(X_val)
        label_names = ["Human", "AI"]
        print("\n  Classification Report (Validation):")
        print(classification_report(y_val, y_pred,
                                    target_names=label_names,
                                    zero_division=0))

    # ── Save model ─────────────────────────────────────
    MODEL_OUTPUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print(f"\n  Model saved → {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
