import requests

url = "http://127.0.0.1:8000/analyse"

print("Testing REAL voice...")
with open("real_voice.mp4", "rb") as f:
    files = {"file": ("real_voice.mp4", f, "audio/mp4")}
    response = requests.post(url, files=files)
    print(response.json())

print("\nTesting CLONED voice...")
with open("cloned_voice.wav", "rb") as f:
    files = {"file": ("cloned_voice.wav", f, "audio/wav")}
    response = requests.post(url, files=files)
    print(response.json())