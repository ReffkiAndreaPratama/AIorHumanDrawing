"""
evaluate.py
Evaluate a trained model on the test split and generate a
detailed performance report.

Usage:
    python src/evaluate.py
"""

import sys
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import load_dataset

MODEL_PATH = Path("model/drawing_detector.joblib")
LABEL_NAMES = ["Human", "AI"]


def plot_confusion_matrix(cm: np.ndarray, output_path: str = "model/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix saved → {output_path}")


def plot_roc_curve(fpr, tpr, auc_score: float,
                   output_path: str = "model/roc_curve.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  ROC curve saved → {output_path}")


def main():
    print("=" * 60)
    print(" AI vs Human Drawing Detector — Evaluation")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model not found at {MODEL_PATH}")
        print("  Run: python src/train.py  first.")
        sys.exit(1)

    print("\n[1/3] Loading model ...")
    model = joblib.load(MODEL_PATH)

    print("[2/3] Loading test data ...")
    X_test, y_test = load_dataset("test")

    if len(X_test) == 0:
        print("\n[WARN] No test data found. "
              "Place images in dataset/test/human/ and dataset/test/ai/")
        sys.exit(1)

    print(f"  {len(X_test)} samples | "
          f"Human: {int(np.sum(y_test == 0))} | "
          f"AI: {int(np.sum(y_test == 1))}")

    print("\n[3/3] Evaluating ...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability for AI class

    acc = np.mean(y_pred == y_test) * 100
    print(f"\n  Test Accuracy: {acc:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=LABEL_NAMES,
                                 zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        print(f"  AUC-ROC: {auc:.4f}")
        plot_roc_curve(fpr, tpr, auc)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
