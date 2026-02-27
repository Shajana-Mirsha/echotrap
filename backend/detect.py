"""
EchoTrap — Voice Clone Detection Engine
Uses trained ML model when available; falls back to rule-based thresholds.
"""

import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import librosa

# ─── Load trained model (if available) ─────────────────────
_ML_DIR    = os.path.join(os.path.dirname(__file__), "..", "ml")
MODEL_PATH  = os.path.join(_ML_DIR, "model.pkl")
SCALER_PATH = os.path.join(_ML_DIR, "scaler.pkl")

_model  = None
_scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  "rb") as f:
            _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        print("[EchoTrap] Trained ML model loaded.")
    else:
        print("[EchoTrap] No trained model found — using rule-based detection.")
except Exception as e:
    print(f"[EchoTrap] Model load error: {e} — falling back to rules.")


# ─── Feature extraction ─────────────────────────────────────
def _extract_features(y, sr):
    """Shared feature extraction for both ML and rule-based modes."""
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr       = librosa.feature.zero_crossing_rate(y)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    rms       = librosa.feature.rms(y=y)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    return {
        "mfcc": mfcc, "zcr": zcr, "centroid": centroid,
        "rolloff": rolloff, "contrast": contrast, "chroma": chroma,
        "rms": rms, "bandwidth": bandwidth,
    }


def _build_feature_vector(feats):
    """Build the 44-dimensional vector matching train_model.py."""
    f = []
    mfcc = feats["mfcc"]
    f.append(float(np.mean(mfcc)))
    f.append(float(np.std(mfcc)))
    for i in range(13):
        f.append(float(np.mean(mfcc[i])))
        f.append(float(np.std(mfcc[i])))

    for key in ["zcr", "centroid", "rolloff", "contrast", "chroma", "rms", "bandwidth"]:
        arr = feats[key]
        f.append(float(np.mean(arr)))
        f.append(float(np.std(arr)))

    return np.array(f, dtype=np.float32).reshape(1, -1)


# ─── Reason generator (human-readable, works for both modes) ─
def _build_reasons(feats, is_clone):
    reasons = []
    if is_clone:
        mfcc_std     = float(np.std(feats["mfcc"]))
        zcr_std      = float(np.std(feats["zcr"]))
        centroid_std = float(np.std(feats["centroid"]))
        contrast_mean = float(np.mean(feats["contrast"]))

        if mfcc_std > 70:
            reasons.append("Unnatural spectral smoothness detected")
        if zcr_std > 0.05:
            reasons.append("Non-human prosody rhythm identified")
        if centroid_std > 300:
            reasons.append("Synthetic tonal pattern in voice signal")
        if contrast_mean > 15:
            reasons.append("Voiceprint mismatch with human baseline")
        if not reasons:
            reasons.append("AI voice fingerprint detected by ML model")
    return reasons


# ─── ML detection ───────────────────────────────────────────
def _detect_ml(y, sr, feats):
    vec    = _build_feature_vector(feats)
    vec_sc = _scaler.transform(vec)
    proba  = _model.predict_proba(vec_sc)[0]   # [p_real, p_clone]
    is_clone = bool(_model.predict(vec_sc)[0])
    confidence = round(float(proba[1]) * 100)

    reasons = _build_reasons(feats, is_clone)
    return {
        "is_clone":         is_clone,
        "confidence_score": confidence,
        "verdict":          "CLONED VOICE DETECTED" if is_clone else "REAL VOICE VERIFIED",
        "reasons":          reasons,
        "method":           "ML Model (RandomForest)",
    }


# ─── Rule-based fallback ────────────────────────────────────
def _detect_rules(feats):
    mfcc_std      = float(np.std(feats["mfcc"]))
    zcr_std       = float(np.std(feats["zcr"]))
    centroid_std  = float(np.std(feats["centroid"]))
    contrast_mean = float(np.mean(feats["contrast"]))

    score   = 0
    reasons = []

    if mfcc_std > 70:
        score += 30
        reasons.append("Unnatural spectral smoothness detected")
    if zcr_std > 0.05:
        score += 30
        reasons.append("Non-human prosody rhythm identified")
    if centroid_std > 300:
        score += 20
        reasons.append("Synthetic tonal pattern in voice signal")
    if contrast_mean > 15:
        score += 20
        reasons.append("Voiceprint mismatch with human baseline")

    is_clone = score >= 50
    return {
        "is_clone":         is_clone,
        "confidence_score": score,
        "verdict":          "CLONED VOICE DETECTED" if is_clone else "REAL VOICE VERIFIED",
        "reasons":          reasons,
        "method":           "Rule-based (no ML model)",
    }


# ─── Public API ─────────────────────────────────────────────
def detect_clone(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=5)
        feats = _extract_features(y, sr)

        if _model is not None and _scaler is not None:
            return _detect_ml(y, sr, feats)
        else:
            return _detect_rules(feats)

    except Exception as e:
        return {
            "is_clone":         False,
            "confidence_score": 0,
            "verdict":          "ERROR",
            "reasons":          [str(e)],
            "method":           "error",
        }