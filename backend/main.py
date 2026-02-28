from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from detect import detect_clone
from alert import send_alert
from pydub import AudioSegment
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend UI at the root
@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "index.html"))


def to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV for librosa. Returns WAV path."""
    if input_path.lower().endswith(".wav"):
        return input_path
    wav_path = input_path.rsplit(".", 1)[0] + ".wav"
    audio = AudioSegment.from_file(input_path)
    audio.export(wav_path, format="wav")
    return wav_path


@app.post("/analyse")
async def analyse_voice(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Auto-convert to WAV (handles MP4, MP3, M4A, OGG, etc.)
    wav_path = to_wav(temp_path)

    result = detect_clone(wav_path)

    if result["is_clone"]:
        send_alert("WARNING. EchoTrap detected a cloned voice.")

    # Clean up all temp files
    for path in set([temp_path, wav_path]):
        if os.path.exists(path):
            os.remove(path)

    return result