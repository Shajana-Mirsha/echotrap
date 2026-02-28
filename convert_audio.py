"""
EchoTrap — Quick Video to WAV Converter
Usage: python convert_audio.py "path\to\your\video.mp4"
Output: creates a .wav file in the same folder you can upload to EchoTrap
"""

import sys
import os
from pydub import AudioSegment

if len(sys.argv) < 2:
    print("Usage: python convert_audio.py <path_to_video_or_audio_file>")
    print("Example: python convert_audio.py C:\\Users\\Me\\Downloads\\voice.mp4")
    sys.exit(1)

input_path = sys.argv[1]

if not os.path.exists(input_path):
    print(f"ERROR: File not found: {input_path}")
    sys.exit(1)

output_path = os.path.splitext(input_path)[0] + "_converted.wav"

print(f"Converting: {input_path}")
audio = AudioSegment.from_file(input_path)
audio.export(output_path, format="wav")
print(f"Saved: {output_path}")
print(f"\nNow upload this file to http://127.0.0.1:8000")
