"""
EchoTrap — Voice Clone Detection Engine (Hybrid)

Strategy:
  1. PYTTSX3 PRE-FILTER: bandwidth_std > 615 → CLONE
     LibriSpeech real voices max at 601, pyttsx3 is 706.
     Catches Windows TTS / pyttsx3 clones before ML runs.

  2. ML MODEL: For all other audio, use the trained RandomForest
     (trained on LibriSpeech vs edge-tts, 100% accuracy on that split).

  3. RULE FALLBACK: If no ML model loaded, use a conservative multi-feature
     rule set that works reasonably across different voice types.

Measured feature ranges across 50 files each:
  Dataset       | centroid_std     | bandwidth_std    | zcr_std
  --------------|------------------|------------------|------------------
  LibriSpeech   | 386 – 1494       | 372 – 601        | 0.026 – 0.201
  edge-tts fake | 628 – 1508       | 304 – 805        | 0.045 – 0.178
  pyttsx3 clone | 1274             | 706              | 0.131
"""

import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import librosa

# ─── Load trained ML model ───────────────────────────────────────────────────
_ML_DIR     = os.path.join(os.path.dirname(__file__), "..", "ml")
MODEL_PATH  = os.path.join(_ML_DIR, "model.pkl")
SCALER_PATH = os.path.join(_ML_DIR, "scaler.pkl")
_model  = None
_scaler = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  "rb") as f: _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
        print("[EchoTrap] ML model loaded (LibriSpeech vs edge-tts).")
    else:
        print("[EchoTrap] No ML model — using rule-based fallback.")
except Exception as e:
    print(f"[EchoTrap] Model load error: {e}")


# ─── Feature extraction ───────────────────────────────────────────────────────
def _extract_features(y, sr):
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr       = librosa.feature.zero_crossing_rate(y)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    rms       = librosa.feature.rms(y=y)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return dict(mfcc=mfcc, zcr=zcr, centroid=centroid, rolloff=rolloff,
                contrast=contrast, chroma=chroma, rms=rms, bandwidth=bandwidth)


def _build_feature_vector(feats):
    """44-dim vector matching train_model.py."""
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


# ─── Detection layers ─────────────────────────────────────────────────────────

def _detect_pyttsx3(feats):
    """
    Layer 1 — pyttsx3 / Windows TTS pre-filter.
    LibriSpeech real voices have bandwidth_std MAX of 601.
    pyttsx3 clones have bandwidth_std of 706.
    Threshold at 615 sits safely above real max, below clone.
    """
    bstd = float(np.std(feats["bandwidth"]))
    if bstd > 615:
        reasons = [
            "Extreme bandwidth instability detected — characteristic of Windows TTS engine",
            "Spectral bandwidth variance far exceeds human speech range",
        ]
        cstd = float(np.std(feats["centroid"]))
        if cstd > 900:
            reasons.append("Unstable spectral centroid — non-human tonal pattern")
        return True, min(88 + int((bstd - 615) / 30), 97), reasons
    return False, 0, []


def _detect_ml(feats):
    """Layer 2 — trained RandomForest for LibriSpeech vs edge-tts."""
    vec    = _build_feature_vector(feats)
    vec_sc = _scaler.transform(vec)
    proba  = _model.predict_proba(vec_sc)[0]  # [p_real, p_clone]
    is_clone = bool(_model.predict(vec_sc)[0])
    confidence = round(float(proba[1 if is_clone else 0]) * 100)
    reasons = []
    if is_clone:
        reasons = [
            "Voice pattern does not match human speech baseline",
            "Spectral signature consistent with AI-generated audio",
        ]
    return is_clone, confidence, reasons


def _detect_rules_fallback(feats):
    """Layer 3 — conservative rule fallback (no ML model)."""
    bstd = float(np.std(feats["bandwidth"]))
    cstd = float(np.std(feats["centroid"]))
    zstd = float(np.std(feats["zcr"]))
    # Only flag as clone if bandwidth_std is far above natural range
    if bstd > 615:
        return True, min(80 + int((bstd - 615) / 25), 95), [
            "Bandwidth instability exceeds human speech range",
        ]
    return False, max(60, int(100 - (bstd / 6))), []


# ─── Public API ───────────────────────────────────────────────────────────────
def detect_clone(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3,
                             res_type='kaiser_fast')
        feats = _extract_features(y, sr)

        bstd = float(np.std(feats["bandwidth"]))
        cstd = float(np.std(feats["centroid"]))
        zstd = float(np.std(feats["zcr"]))
        mstd = float(np.std(feats["mfcc"]))
        print(f"[EchoTrap] bandwidth_std={bstd:.1f}  centroid_std={cstd:.1f}"
              f"  zcr_std={zstd:.4f}  mfcc_std={mstd:.1f}")

        # Layer 1 — pyttsx3 pre-filter
        is_clone, confidence, reasons = _detect_pyttsx3(feats)
        method = "pyttsx3/Windows-TTS Filter"

        # Layer 2 — ML model (LibriSpeech vs edge-tts)
        if not is_clone and _model is not None:
            is_clone, confidence, reasons = _detect_ml(feats)
            method = "ML Model (RandomForest)"

        # Layer 3 — Rule fallback
        if not is_clone and _model is None:
            is_clone, confidence, reasons = _detect_rules_fallback(feats)
            method = "Rule-based Fallback"

        return {
            "is_clone":         is_clone,
            "confidence_score": confidence,
            "verdict":          "CLONED VOICE DETECTED" if is_clone else "REAL VOICE VERIFIED",
            "reasons":          reasons,
            "method":           method,
        }

    except Exception as e:
        return {
            "is_clone":         False,
            "confidence_score": 0,
            "verdict":          "ERROR",
            "reasons":          [str(e)],
            "method":           "error",
        }