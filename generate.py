"""
Text-to-Speech using MMS-TTS (Offline Setup)

INSTALLATION:
1. Install dependencies:
   pip install transformers torch scipy numpy

2. Download model to local folder:
   python -c "from transformers import VitsModel, AutoTokenizer; model = VitsModel.from_pretrained('facebook/mms-tts-eng'); tokenizer = AutoTokenizer.from_pretrained('facebook/mms-tts-eng'); model.save_pretrained('model_tts'); tokenizer.save_pretrained('model_tts'); print('Model saved to model_tts folder')"

3. Run script (will use local model):
   python generate.py

Model size: ~400MB
Location: model_tts/ folder
"""

import torch
from transformers import VitsModel, AutoTokenizer

# ------------------ CONFIG ------------------
TEXT = "Hello! This is a test of text to speech conversion. It works amazingly well with minimal setup."
OUTPUT_AUDIO = "output_speech.wav"
LANGUAGE = "en"  # Language code
SPEAKER_ID = 0  # Voice variant (0-10 for different voices)
# -------------------------------------------

print("Loading VITS text-to-speech model...")
# Load model and tokenizer from local folder
model = VitsModel.from_pretrained("model_tts", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("model_tts", local_files_only=True)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Generating speech on {device}...")
# Tokenize text
inputs = tokenizer(TEXT, return_tensors="pt").to(device)

# Generate speech
with torch.no_grad():
    output = model(**inputs).waveform

# Save audio file
import scipy.io.wavfile as wavfile
import numpy as np

# Convert to numpy and save
audio_data = output.squeeze().cpu().numpy()
sample_rate = model.config.sampling_rate

# Normalize audio to int16 range
audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

wavfile.write(OUTPUT_AUDIO, rate=sample_rate, data=audio_data)

print(f"Audio saved → {OUTPUT_AUDIO}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
