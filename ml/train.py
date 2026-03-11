import os
import json
import numpy as np
from features import extract_mfcc, extract_spectral, extract_prosody, extract_voiceprint

def extract_all_features(audio_path):
    mfcc = extract_mfcc(audio_path)
    spectral = extract_spectral(audio_path)
    prosody = extract_prosody(audio_path)
    voiceprint = extract_voiceprint(audio_path)
    
    return {
        **mfcc,
        **spectral,
        **prosody,
        **voiceprint
    }

def train_baseline(real_audio_folder):
    baselines = []
    
    for file in os.listdir(real_audio_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            path = os.path.join(real_audio_folder, file)
            features = extract_all_features(path)
            baselines.append(features)
    
    if not baselines:
        print("No audio files found in folder")
        return None
    
    # Calculate average baseline from real voices
    baseline = {}
    for key in baselines[0].keys():
        baseline[key] = float(np.mean([b[key] for b in baselines]))
    
    # Save baseline to file
    with open("baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    
    print("Baseline trained and saved successfully")
    print(json.dumps(baseline, indent=2))
    return baseline

if __name__ == "__main__":
    train_baseline("real_voices")
