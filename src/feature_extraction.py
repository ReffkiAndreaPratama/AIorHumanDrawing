"""
feature_extraction.py
Extracts classical computer vision features from preprocessed images.

Feature groups:
  1. GLCM (Gray-Level Co-occurrence Matrix) — texture descriptor
  2. Edge features — edge density, gradient statistics
  3. Frequency domain — FFT-based smoothness measure
  4. Statistical features — entropy, local variance
  5. Color histogram — hue/saturation distribution
  6. Sharpness & noise estimation

All features are concatenated into a single 1-D float32 vector
that is fed into the classifier.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from scipy.stats import entropy as scipy_entropy


# ─────────────────────────────────────────────
# 1. GLCM features
# ─────────────────────────────────────────────

def extract_glcm_features(gray: np.ndarray) -> dict:
    """
    Compute GLCM-based texture features.

    Parameters
    ----------
    gray : uint8 grayscale image

    Returns
    -------
    dict with keys: contrast, dissimilarity, homogeneity,
                    energy, correlation, ASM
    """
    # Quantize to 64 gray levels to keep computation manageable
    img_q = (gray // 4).astype(np.uint8)

    # Compute GLCM at 4 angles, distance=1
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(img_q, distances=distances, angles=angles,
                        levels=64, symmetric=True, normed=True)

    props = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy",
                 "correlation", "ASM"]:
        values = graycoprops(glcm, prop).flatten()
        props[f"glcm_{prop}_mean"] = float(np.mean(values))
        props[f"glcm_{prop}_std"] = float(np.std(values))

    return props


# ─────────────────────────────────────────────
# 2. Edge features
# ─────────────────────────────────────────────

def extract_edge_features(gray: np.ndarray) -> dict:
    """
    Compute edge density and gradient statistics.

    AI-generated drawings often have very clean, uniform edges.
    Human drawings tend to have more irregular, variable strokes.
    """
    gray_f = gray.astype(np.float32)

    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / float(edges.size)

    # Sobel gradient magnitude
    sobel_img = sobel(gray_f / 255.0)
    grad_mean = float(np.mean(sobel_img))
    grad_std = float(np.std(sobel_img))
    grad_max = float(np.max(sobel_img))

    # Laplacian (sharpness proxy)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    return {
        "edge_density": edge_density,
        "gradient_mean": grad_mean,
        "gradient_std": grad_std,
        "gradient_max": grad_max,
        "laplacian_variance": laplacian_var,
    }


# ─────────────────────────────────────────────
# 3. Frequency domain features
# ─────────────────────────────────────────────

def extract_frequency_features(gray: np.ndarray) -> dict:
    """
    FFT-based measure of high-frequency content.

    AI images tend to have distinct high-frequency spectral patterns
    compared to scanned/photographed human drawings.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)
    log_magnitude = np.log1p(magnitude)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4

    # Create masks for low and high frequency rings
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    low_freq_energy = float(np.sum(log_magnitude[dist <= radius]))
    high_freq_energy = float(np.sum(log_magnitude[dist > radius]))
    total_energy = low_freq_energy + high_freq_energy + 1e-9

    return {
        "fft_low_freq_ratio": low_freq_energy / total_energy,
        "fft_high_freq_ratio": high_freq_energy / total_energy,
        "fft_energy_total": float(np.sum(log_magnitude)),
    }


# ─────────────────────────────────────────────
# 4. Statistical / entropy features
# ─────────────────────────────────────────────

def extract_statistical_features(gray: np.ndarray) -> dict:
    """
    Pixel-level statistics and entropy measures.
    """
    flat = gray.flatten().astype(np.float32)

    # Pixel histogram entropy
    hist, _ = np.histogram(flat, bins=256, range=(0, 256), density=True)
    hist = hist + 1e-9  # avoid log(0)
    pixel_entropy = float(scipy_entropy(hist, base=2))

    # Local variance via sliding window (block-based)
    block_size = 16
    h, w = gray.shape
    local_vars = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            local_vars.append(float(np.var(block)))

    local_var_mean = float(np.mean(local_vars)) if local_vars else 0.0
    local_var_std = float(np.std(local_vars)) if local_vars else 0.0

    return {
        "pixel_entropy": pixel_entropy,
        "pixel_mean": float(np.mean(flat)),
        "pixel_std": float(np.std(flat)),
        "local_variance_mean": local_var_mean,
        "local_variance_std": local_var_std,
    }


# ─────────────────────────────────────────────
# 5. Color histogram features
# ─────────────────────────────────────────────

def extract_color_features(color_bgr: np.ndarray) -> dict:
    """
    HSV histogram statistics.
    AI images often have unusually saturated or perfectly smooth gradients.

    Parameters
    ----------
    color_bgr : uint8 BGR image (resized, not normalized)
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

    features = {}
    channel_names = ["hue", "saturation", "value"]
    for i, name in enumerate(channel_names):
        ch = hsv[:, :, i].flatten().astype(np.float32)
        features[f"color_{name}_mean"] = float(np.mean(ch))
        features[f"color_{name}_std"] = float(np.std(ch))
        hist, _ = np.histogram(ch, bins=32, density=True)
        hist = hist + 1e-9
        features[f"color_{name}_entropy"] = float(scipy_entropy(hist, base=2))

    return features


# ─────────────────────────────────────────────
# 6. Noise estimation
# ─────────────────────────────────────────────

def estimate_noise(gray: np.ndarray) -> dict:
    """
    Estimate noise level using a high-pass filter residual.

    Human drawings (especially scanned ones) often have paper grain.
    AI images can have either zero noise or synthetic noise patterns.
    """
    # Gaussian blur then subtract to isolate noise
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = gray.astype(np.float32) - blurred
    noise_std = float(np.std(residual))
    noise_mean_abs = float(np.mean(np.abs(residual)))

    return {
        "noise_std": noise_std,
        "noise_mean_abs": noise_mean_abs,
    }


# ─────────────────────────────────────────────
# Master extractor
# ─────────────────────────────────────────────

def extract_all_features(gray: np.ndarray,
                         color_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run all feature extractors and return:
      - feature_vector : 1-D float32 numpy array (for model input)
      - feature_dict   : dict with named values (for UI display)

    Parameters
    ----------
    gray      : uint8 grayscale image (256×256)
    color_bgr : uint8 BGR color image (256×256)
    """
    feature_dict: dict = {}

    feature_dict.update(extract_glcm_features(gray))
    feature_dict.update(extract_edge_features(gray))
    feature_dict.update(extract_frequency_features(gray))
    feature_dict.update(extract_statistical_features(gray))
    feature_dict.update(extract_color_features(color_bgr))
    feature_dict.update(estimate_noise(gray))

    # Build a fixed-order vector
    feature_vector = np.array(list(feature_dict.values()), dtype=np.float32)

    return feature_vector, feature_dict


def get_display_features(feature_dict: dict) -> dict:
    """
    Return a curated subset of features to show in the UI.
    Keys are human-readable labels.
    """
    mapping = {
        "Edge Density":       feature_dict.get("edge_density", 0),
        "Texture Energy":     feature_dict.get("glcm_energy_mean", 0),
        "Contrast":           feature_dict.get("glcm_contrast_mean", 0),
        "Homogeneity":        feature_dict.get("glcm_homogeneity_mean", 0),
        "Sharpness":          feature_dict.get("laplacian_variance", 0),
        "Pixel Entropy":      feature_dict.get("pixel_entropy", 0),
        "Noise Level":        feature_dict.get("noise_std", 0),
        "Gradient Std":       feature_dict.get("gradient_std", 0),
    }
    return {k: round(float(v), 4) for k, v in mapping.items()}
