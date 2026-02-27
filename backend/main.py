from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from detect import detect_clone
from alert import send_alert
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "EchoTrap is running"}

@app.post("/analyse")
async def analyse_voice(file: UploadFile = File(...)):
    
    # Save uploaded audio temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Run detection
    result = detect_clone(temp_path)
    
    # Send alert if clone detected
    if result["is_clone"]:
        send_alert("WARNING. EchoTrap detected a cloned voice on your family members phone. Check on them immediately.")
    
    # Clean up temp file
    os.remove(temp_path)
    
    return result