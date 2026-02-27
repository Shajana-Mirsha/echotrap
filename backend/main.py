from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from detect import detect_clone
from alert import send_alert
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

@app.post("/analyse")
async def analyse_voice(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    result = detect_clone(temp_path)
    
    if result["is_clone"]:
        send_alert("WARNING. EchoTrap detected a cloned voice.")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return result