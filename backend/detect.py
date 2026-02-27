import librosa
import numpy as np

def detect_clone(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=5)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        mfcc_std = float(np.std(mfcc))
        zcr_std = float(np.std(zcr))
        centroid_std = float(np.std(centroid))
        contrast_mean = float(np.mean(contrast))
        
        clone_score = 0
        reasons = []

        if mfcc_std > 70:
            clone_score += 30
            reasons.append("Unnatural voice frequency pattern")

        if zcr_std > 0.05:
            clone_score += 30
            reasons.append("Abnormal rhythm consistency")

        if centroid_std > 300:
            clone_score += 20
            reasons.append("Synthetic spectral pattern")

        if contrast_mean > 15:
            clone_score += 20
            reasons.append("Unnatural tonal contrast")

        is_clone = clone_score >= 50

        return {
            "is_clone": is_clone,
            "confidence_score": clone_score,
            "verdict": "CLONED VOICE DETECTED" if is_clone else "REAL VOICE VERIFIED",
            "reasons": reasons
        }

    except Exception as e:
        return {
            "is_clone": False,
            "confidence_score": 0,
            "verdict": "ERROR",
            "reasons": [str(e)]
        }