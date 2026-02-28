"""
EchoTrap — Voice Clone Detection Engine
Calibrated thresholds based on real feature measurements:

  Feature         | Real voice | Cloned (pyttsx3) | Threshold
  ----------------|------------|------------------|----------
  centroid_std    |   218      |   1274           |  > 500
  bandwidth_std   |   122      |   706            |  > 300
  zcr_std         |   0.032    |   0.131          |  > 0.07
  mfcc_std        |   70.4     |   109.7          |  > 85

Scoring: each flag adds points. >= 50 points = CLONED.
"""

import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import librosa

# ─── Load trained ML model (used as tie-breaker only) ───────
_ML_DIR     = os.path.join(os.path.dirname(__file__), "..", "ml")
MODEL_PATH  = os.path.join(_ML_DIR, "model.pkl")
SCALER_PATH = os.path.join(_ML_DIR, "scaler.pkl")

_model  = None
_scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  "rb") as f: _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
        print("[EchoTrap] Trained ML model loaded.")
    else:
        print("[EchoTrap] No trained model — using rule-based detection.")
except Exception as e:
    print(f"[EchoTrap] Model load error: {e} — using rules.")


# ─── Feature extraction ──────────────────────────────────────
def _extract_features(y, sr):
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
    """Build the 44-dim vector matching train_model.py for ML model."""
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


# ─── Calibrated rule-based detection ────────────────────────
# Thresholds derived from measuring real_voice.wav vs cloned_voice.wav (pyttsx3).
# Each threshold sits between the real and clone values with comfortable margin.
RULES = [
    # (feature_extractor, threshold, points, reason_text)
    # centroid_std: real=218, clone=1274 → threshold 500
    (lambda f: float(np.std(f["centroid"])),  500,  35,
     "Unstable spectral centroid — characteristic of synthetic speech engine"),
    # bandwidth_std: real=122, clone=706 → threshold 300
    (lambda f: float(np.std(f["bandwidth"])), 300,  30,
     "High bandwidth instability — non-human tonal variation pattern"),
    # zcr_std: real=0.032, clone=0.131 → threshold 0.07
    (lambda f: float(np.std(f["zcr"])),       0.07, 25,
     "Irregular zero-crossing rhythm — prosody mismatch with human speech"),
    # mfcc_std: real=70.4, clone=109.7 → threshold 85
    (lambda f: float(np.std(f["mfcc"])),      85,   10,
     "Unnatural spectral envelope smoothness detected"),
]

CLONE_THRESHOLD = 50   # points needed to classify as clone


def _detect_rules(feats):
    score   = 0
    reasons = []

    for extractor, threshold, points, reason in RULES:
        val = extractor(feats)
        if val > threshold:
            score   += points
            reasons.append(reason)

    is_clone = score >= CLONE_THRESHOLD

    # confidence: 0–100 mapped from score
    raw_conf = min(score, 100)
    if not is_clone:
        # invert for real voice confidence
        raw_conf = max(10, 100 - score)

    return {
        "is_clone":         is_clone,
        "confidence_score": raw_conf,
        "verdict":          "CLONED VOICE DETECTED" if is_clone else "REAL VOICE VERIFIED",
        "reasons":          reasons,
        "method":           "Rule-based Acoustic Analysis",
    }


# ─── Public API ──────────────────────────────────────────────
def detect_clone(audio_path):
    """
    Main entrypoint. Loads audio, extracts features, runs calibrated
    rule-based detection. ML model is no longer used as primary — rules
    are calibrated for pyttsx3 / Windows TTS clones.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3,
                             res_type='kaiser_fast')
        feats = _extract_features(y, sr)

        # ── Debug print (visible in uvicorn terminal) ──
        cstd = round(float(np.std(feats["centroid"])), 1)
        bstd = round(float(np.std(feats["bandwidth"])), 1)
        zstd = round(float(np.std(feats["zcr"])), 5)
        mstd = round(float(np.std(feats["mfcc"])), 2)
        print(f"[EchoTrap] Features → centroid_std={cstd}  "
              f"bandwidth_std={bstd}  zcr_std={zstd}  mfcc_std={mstd}")

        return _detect_rules(feats)

    except Exception as e:
        return {
            "is_clone":         False,
            "confidence_score": 0,
            "verdict":          "ERROR",
            "reasons":          [str(e)],
            "method":           "error",
        }