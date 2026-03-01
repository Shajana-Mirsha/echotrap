# EchoTrap - AI Voice Clone Detector

> Real-time AI-powered protection against synthetic voice fraud, built to protect elderly people from scammers who clone their children's voices.

---

## The Problem

AI voice cloning technology can now replicate any person's voice in under 3 seconds using a 10-second audio clip from anywhere - a phone call, a voicemail, a video. Scammers are using this to call elderly people in the voices of their own children, creating fake emergencies and demanding immediate money transfers.

**The victim hears a voice they have known and loved for decades. They do not question it.**

- 140 million elderly people in India alone
- 3.4 billion people above 50 globally are at risk
- Voice cloning scams caused over $25 billion in losses globally in 2024
- **No real-time consumer solution existed. Until now.**

---

## The Solution

EchoTrap is an AI-powered web application that analyses any incoming audio and detects AI-generated synthetic voices in real time using a **3-layer detection engine**:

| Layer | Method | What it catches |
|---|---|---|
| Layer 1 | MFCC spectral analysis | Unnatural voice frequency patterns |
| Layer 2 | Zero-crossing rate deviation | Abnormal rhythm consistency |
| Layer 3 | Spectral centroid + contrast | Synthetic tonal patterns |

If 2+ layers flag an anomaly → **CLONED VOICE DETECTED** alert triggered within 4 seconds.

---

## Demo

Upload any audio file to detect whether it is a real human voice or an AI-generated clone.

**Real voice:** ✔ REAL VOICE VERIFIED (green UI)  
**Cloned voice:** ⚠ CLONED VOICE DETECTED (red UI, with specific reasons)

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.13, FastAPI, Uvicorn |
| AI Engine | Librosa, NumPy (MFCC, ZCR, Spectral Analysis) |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Alert System | File-based logging (Twilio-ready) |

---

## Project Structure

```
echotrap/
├── index.html          # Frontend UI (served by FastAPI)
├── backend/
│   ├── main.py         # FastAPI server + routes
│   ├── detect.py       # 3-layer AI detection engine
│   ├── alert.py        # Alert system
│   └── requirements.txt
└── ml/
    ├── features.py     # Feature extraction utilities
    └── train.py        # Training pipeline (for future dataset training)
```

---

## Setup & Run

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn main:app --reload
```

### 3. Open the app

Visit **http://127.0.0.1:8000** in your browser.

---

## How Detection Works

```
Incoming audio
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 1: MFCC Std Dev > 70?  → clone_score += 30  │
│  Layer 2: ZCR Std Dev > 0.05? → clone_score += 30  │
│  Layer 3: Centroid Std > 300? → clone_score += 20  │
│           Contrast Mean > 15? → clone_score += 20  │
└─────────────────────────────────────────────────────┘
     │
     ▼
  score ≥ 50 → CLONED VOICE DETECTED ⚠
  score < 50 → REAL VOICE VERIFIED ✔
```

Real human voices have **natural biological variation** — micro-tremors, breath patterns, irregular rhythm. AI-generated voices are **mathematically smooth** — too consistent, with unnatural spectral patterns. EchoTrap measures exactly this difference.

---




## License

MIT