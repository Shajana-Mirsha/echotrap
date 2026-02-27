import librosa
import numpy as np

def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "mean": float(np.mean(mfcc)),
        "std": float(np.std(mfcc)),
        "max": float(np.max(mfcc)),
        "min": float(np.min(mfcc))
    }

def extract_spectral(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return {
        "rolloff_mean": float(np.mean(rolloff)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth))
    }

def extract_prosody(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rmse_mean": float(np.mean(rmse)),
        "tempo": float(tempo)
    }

def extract_voiceprint(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return {
        "chroma_mean": float(np.mean(chroma)),
        "chroma_std": float(np.std(chroma)),
        "contrast_mean": float(np.mean(contrast))
    }