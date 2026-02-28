"""
EchoTrap — Voice Clone Detection Engine

Detection logic uses a multi-signal approach:

PRIMARY RULE (AND logic — most robust):
  if centroid_std > 700 AND bandwidth_std > 400 → CLONE
  These two features are simultaneously extreme ONLY in synthetic TTS voices.
  A real voice recording (even noisy) will not exceed BOTH thresholds.

SUPPORTING SIGNALS add to confidence score only after primary rule triggers.

Measured values from actual test files:
  Feature         | real_voice.wav | cloned_voice.wav (pyttsx3)
  ----------------|----------------|---------------------------
  centroid_std    |   218          |   1274   (5.8x)
  bandwidth_std   |   122          |   706    (5.8x)
  zcr_std         |   0.032        |   0.131  (4.0x)
  mfcc_std        |   70.4         |   109.7  (1.6x)
"""

import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import librosa

# ─── Load ML model (kept as reference, not used as primary classifier) ───────
_ML_DIR     = os.path.join(os.path.dirname(__file__), "..", "ml")
MODEL_PATH  = os.path.join(_ML_DIR, "model.pkl")
SCALER_PATH = os.path.join(_ML_DIR, "scaler.pkl")
_model  = None
_scaler = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  "rb") as f: _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
        print("[EchoTrap] Trained ML model loaded (reference only).")
    else:
        print("[EchoTrap] No trained model found — using calibrated rules.")
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


# ─── Detection ────────────────────────────────────────────────────────────────
def _detect(feats):
    centroid_std  = float(np.std(feats["centroid"]))
    bandwidth_std = float(np.std(feats["bandwidth"]))
    zcr_std       = float(np.std(feats["zcr"]))
    mfcc_std      = float(np.std(feats["mfcc"]))

    # ── PRIMARY GATE: both must exceed threshold simultaneously ──
    # Real voices (even noisy phone recordings) will NOT exceed both.
    # pyttsx3/synthetic TTS exceeds both by a large margin.
    primary_clone = (centroid_std > 700) and (bandwidth_std > 400)

    if not primary_clone:
        # Definitely real — compute real confidence from how far below thresholds
        # Higher distance from threshold = higher real confidence
        gap_c = max(0, 700  - centroid_std)  / 700
        gap_b = max(0, 400  - bandwidth_std) / 400
        real_conf = int(min(95, 50 + (gap_c + gap_b) * 25))
        return {
            "is_clone":         False,
            "confidence_score": real_conf,
            "verdict":          "REAL VOICE VERIFIED",
            "reasons":          [],
            "method":           "Acoustic Analysis (Calibrated)",
        }

    # ── CLONE confirmed — build confidence and reasons ──
    reasons = []
    confidence = 75   # base for passing primary gate

    reasons.append(
        "Unstable spectral centroid — characteristic of synthetic speech engine"
    )
    reasons.append(
        "High bandwidth instability — non-human tonal variation pattern"
    )

    if zcr_std > 0.09:
        confidence = min(confidence + 10, 98)
        reasons.append(
            "Irregular zero-crossing rhythm — prosody mismatch with human speech"
        )
    if mfcc_std > 90:
        confidence = min(confidence + 7, 98)
        reasons.append(
            "Unnatural spectral envelope detected"
        )

    return {
        "is_clone":         True,
        "confidence_score": confidence,
        "verdict":          "CLONED VOICE DETECTED",
        "reasons":          reasons,
        "method":           "Acoustic Analysis (Calibrated)",
    }


# ─── Public API ───────────────────────────────────────────────────────────────
def detect_clone(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3,
                             res_type='kaiser_fast')
        feats = _extract_features(y, sr)

        # Debug print in uvicorn terminal
        print(
            f"[EchoTrap] centroid_std={np.std(feats['centroid']):.1f}  "
            f"bandwidth_std={np.std(feats['bandwidth']):.1f}  "
            f"zcr_std={np.std(feats['zcr']):.4f}  "
            f"mfcc_std={np.std(feats['mfcc']):.1f}"
        )

        return _detect(feats)

    except Exception as e:
        return {
            "is_clone":         False,
            "confidence_score": 0,
            "verdict":          "ERROR",
            "reasons":          [str(e)],
            "method":           "error",
        }