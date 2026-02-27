"""
EchoTrap — Dataset Preparation
Downloads LibriSpeech test-clean (real voices, no login needed)
and generates AI fake voices using edge-tts.
"""

import os
import asyncio
import glob
import shutil
import urllib.request
import tarfile

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
REAL_DIR   = os.path.join(DATA_DIR, "real")
FAKE_DIR   = os.path.join(DATA_DIR, "fake")

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
MAX_SAMPLES = 150   # 150 real + 150 fake = 300 training samples

# edge-tts voices to rotate through (adds diversity to fake samples)
FAKE_VOICES = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
]

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


# ─── Download ──────────────────────────────────────────────
def show_progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 // total_size, 100)
        print(f"\r  Downloading… {pct}%   ", end="", flush=True)


def download_librispeech():
    tar_path       = os.path.join(DATA_DIR, "test-clean.tar.gz")
    extracted_dir  = os.path.join(DATA_DIR, "LibriSpeech")

    if os.path.exists(extracted_dir):
        print("  LibriSpeech already downloaded — skipping.")
        return

    if not os.path.exists(tar_path):
        print("Downloading LibriSpeech test-clean (~346 MB)…")
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path, show_progress)
        print()

    print("  Extracting…")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("  Done.")


# ─── Collect samples ───────────────────────────────────────
def collect_samples():
    librispeech_dir = os.path.join(DATA_DIR, "LibriSpeech", "test-clean")
    samples = []

    for speaker in sorted(os.listdir(librispeech_dir)):
        speaker_path = os.path.join(librispeech_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue

        for chapter in sorted(os.listdir(speaker_path)):
            chapter_path = os.path.join(speaker_path, chapter)
            if not os.path.isdir(chapter_path):
                continue

            # Read transcripts
            transcripts = {}
            for tf in glob.glob(os.path.join(chapter_path, "*.txt")):
                with open(tf, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]

            for flac_path in sorted(glob.glob(os.path.join(chapter_path, "*.flac"))):
                file_id = os.path.splitext(os.path.basename(flac_path))[0]
                if file_id in transcripts:
                    samples.append({
                        "id":    file_id,
                        "audio": flac_path,
                        "text":  transcripts[file_id],
                    })
                    if len(samples) >= MAX_SAMPLES:
                        return samples
    return samples


def copy_real_audio(samples):
    copied = 0
    for s in samples:
        dest = os.path.join(REAL_DIR, f"{s['id']}.flac")
        if not os.path.exists(dest):
            shutil.copy2(s["audio"], dest)
            copied += 1
    print(f"  Copied {copied} new real audio files (total {len(samples)}).")


# ─── Generate fake voices ──────────────────────────────────
async def _gen_one(text, output_path, voice):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def generate_all_fakes(samples):
    tasks = []
    for i, s in enumerate(samples):
        output_path = os.path.join(FAKE_DIR, f"{s['id']}.mp3")
        if not os.path.exists(output_path):
            voice = FAKE_VOICES[i % len(FAKE_VOICES)]
            tasks.append(_gen_one(s["text"], output_path, voice))

    if not tasks:
        print("  All fake voices already generated — skipping.")
        return

    print(f"  Generating {len(tasks)} fake voice samples…")
    BATCH = 8
    for i in range(0, len(tasks), BATCH):
        batch = tasks[i : i + BATCH]
        await asyncio.gather(*batch)
        done = min(i + BATCH, len(tasks))
        print(f"    {done}/{len(tasks)}", end="\r", flush=True)
    print(f"  All {len(tasks)} fake voices generated.        ")


# ─── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  EchoTrap — Dataset Preparation")
    print("=" * 50)

    download_librispeech()

    print("\nCollecting real voice samples…")
    samples = collect_samples()
    print(f"  Found {len(samples)} samples.")

    print("\nCopying real audio files…")
    copy_real_audio(samples)

    print("\nGenerating AI fake voices (edge-tts)…")
    asyncio.run(generate_all_fakes(samples))

    real_count = len(glob.glob(os.path.join(REAL_DIR, "*.flac")))
    fake_count = len(glob.glob(os.path.join(FAKE_DIR, "*.mp3")))

    print(f"\n✓ Dataset ready:")
    print(f"    Real voices : {real_count}  ({REAL_DIR})")
    print(f"    Fake voices : {fake_count}  ({FAKE_DIR})")
    print("\nNext step → run: python train_model.py")
