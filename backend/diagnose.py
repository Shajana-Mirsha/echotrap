"""
EchoTrap Diagnostic — prints all feature values for real_voice.wav and cloned_voice.wav
Run from the backend/ directory: python diagnose.py
"""
import librosa
import numpy as np

files = {
    "REAL  (real_voice.wav)":   "real_voice.wav",
    "CLONE (cloned_voice.wav)": "cloned_voice.wav",
}

for label, path in files.items():
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        y, sr = librosa.load(path, sr=16000, duration=3, res_type='kaiser_fast')
        print(f"  Samples: {len(y)}  |  Sample rate: {sr}")

        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr       = librosa.feature.zero_crossing_rate(y)
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
        contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)
        chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
        rms       = librosa.feature.rms(y=y)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        print(f"\n  MFCC      mean={np.mean(mfcc):.4f}  std={np.std(mfcc):.4f}  max={np.max(mfcc):.4f}  min={np.min(mfcc):.4f}")
        for i in range(13):
            print(f"    MFCC[{i:02d}]  mean={np.mean(mfcc[i]):.4f}  std={np.std(mfcc[i]):.4f}")
        print(f"\n  ZCR       mean={np.mean(zcr):.6f}  std={np.std(zcr):.6f}")
        print(f"  Centroid  mean={np.mean(centroid):.2f}   std={np.std(centroid):.2f}")
        print(f"  Rolloff   mean={np.mean(rolloff):.2f}   std={np.std(rolloff):.2f}")
        print(f"  Contrast  mean={np.mean(contrast):.4f}  std={np.std(contrast):.4f}")
        print(f"  Chroma    mean={np.mean(chroma):.4f}  std={np.std(chroma):.4f}")
        print(f"  RMS       mean={np.mean(rms):.6f}  std={np.std(rms):.6f}")
        print(f"  Bandwidth mean={np.mean(bandwidth):.2f}   std={np.std(bandwidth):.2f}")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print("  Diagnostic complete.")
print(f"{'='*60}\n")
