"""
EchoTrap — Model Training
Extracts audio features from real/fake dataset and trains a RandomForest classifier.
Saves model.pkl + scaler.pkl to ml/ directory.
"""

import os
import glob
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import librosa
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
REAL_DIR    = os.path.join(DATA_DIR, "real")
FAKE_DIR    = os.path.join(DATA_DIR, "fake")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")


# ─── Feature Extraction ────────────────────────────────────
def extract_features(audio_path):
    """
    Extract 44-dimensional feature vector from audio file.
    Same features used by detect.py for consistency.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=5)

        if len(y) < 2000:
            return None

        features = []

        # 1) MFCC — 13 coefficients × (mean + std) = 26 features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.append(float(np.mean(mfcc)))
        features.append(float(np.std(mfcc)))
        for i in range(13):
            features.append(float(np.mean(mfcc[i])))
            features.append(float(np.std(mfcc[i])))

        # 2) Zero-Crossing Rate — mean + std = 2
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # 3) Spectral Centroid — mean + std = 2
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))

        # 4) Spectral Rolloff — mean + std = 2
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(float(np.mean(rolloff)))
        features.append(float(np.std(rolloff)))

        # 5) Spectral Contrast — mean + std = 2
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.append(float(np.mean(contrast)))
        features.append(float(np.std(contrast)))

        # 6) Chroma — mean + std = 2
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(float(np.mean(chroma)))
        features.append(float(np.std(chroma)))

        # 7) RMS Energy — mean + std = 2
        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        # 8) Spectral Bandwidth — mean + std = 2
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        return None


def process_directory(directory, label, extensions):
    """Extract features from all audio files in directory."""
    X, y = [], []
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))

    label_name = "REAL" if label == 0 else "FAKE"
    print(f"  [{label_name}] processing {len(files)} files…")

    errors = 0
    for i, path in enumerate(files):
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(label)
        else:
            errors += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            print(f"    {i+1}/{len(files)} done ({errors} skipped)", end="\r")

    print(f"    {len(files)} done, {errors} skipped         ")
    return X, y


# ─── Training ──────────────────────────────────────────────
def train():
    print("=" * 50)
    print("  EchoTrap — Model Training")
    print("=" * 50)

    # Check dataset exists
    if not os.path.exists(REAL_DIR) or not os.path.exists(FAKE_DIR):
        print("ERROR: Dataset not found. Run prepare_data.py first.")
        return

    print("\nExtracting features from real voices…")
    X_real, y_real = process_directory(REAL_DIR, label=0, extensions=["flac", "wav"])

    print("\nExtracting features from fake voices…")
    X_fake, y_fake = process_directory(FAKE_DIR, label=1, extensions=["mp3", "wav"])

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    print(f"\nDataset summary:")
    print(f"  Real samples : {sum(y == 0)}")
    print(f"  Fake samples : {sum(y == 1)}")
    print(f"  Feature dims : {X.shape[1]}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train RandomForest
    print("\nTraining RandomForest classifier…")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY : {acc:.1%}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Clone"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:")
    print(f"  True Real  caught: {cm[0][0]}  |  Real misclassified as Clone: {cm[0][1]}")
    print(f"  Clone caught     : {cm[1][1]}  |  Clone missed                : {cm[1][0]}")

    # Save
    with open(MODEL_PATH,  "wb") as f:
        pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✓ Model saved  → {MODEL_PATH}")
    print(f"✓ Scaler saved → {SCALER_PATH}")
    print("\nNext step → restart backend and test at http://127.0.0.1:8000")

    return acc


if __name__ == "__main__":
    train()
