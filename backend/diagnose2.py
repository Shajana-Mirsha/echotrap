import librosa, numpy as np, json

result = {}
for label, path in [("real", "real_voice.wav"), ("clone", "cloned_voice.wav")]:
    y, sr = librosa.load(path, sr=16000, duration=3, res_type='kaiser_fast')
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr      = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    rms      = librosa.feature.rms(y=y)
    bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=sr)
    result[label] = {
        "mfcc_mean": round(float(np.mean(mfcc)),4),
        "mfcc_std":  round(float(np.std(mfcc)),4),
        "zcr_mean":  round(float(np.mean(zcr)),6),
        "zcr_std":   round(float(np.std(zcr)),6),
        "centroid_mean": round(float(np.mean(centroid)),2),
        "centroid_std":  round(float(np.std(centroid)),2),
        "rolloff_mean":  round(float(np.mean(rolloff)),2),
        "contrast_mean": round(float(np.mean(contrast)),4),
        "contrast_std":  round(float(np.std(contrast)),4),
        "chroma_mean":   round(float(np.mean(chroma)),4),
        "rms_mean":      round(float(np.mean(rms)),6),
        "bandwidth_mean":round(float(np.mean(bandwidth)),2),
        "bandwidth_std": round(float(np.std(bandwidth)),2),
    }

with open("features.json","w") as f:
    json.dump(result, f, indent=2)
print("done")
