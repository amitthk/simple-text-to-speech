"""
Text-to-Speech using MMS-TTS (Offline Setup)

INSTALLATION:
1. Install dependencies:
   pip install transformers torch scipy numpy huggingface_hub

2. Download model to local folder:
   python download_hf_minimal.py --repo-id facebook/mms-tts-eng --output-dir model_tts

3. Run script (will use local model):
   python generate.py --text-file togaf.txt

Model size: ~400MB
Location: model_tts/ folder
"""

import argparse
import hashlib
import re
from pathlib import Path

import torch
from transformers import VitsModel, AutoTokenizer

# ------------------ CONFIG ------------------
LANGUAGE = "en"  # Language code
SPEAKER_ID = 0  # Voice variant (0-10 for different voices)
MAX_CHARS_PER_CHUNK = 600  # Reduce memory use for long inputs
# -------------------------------------------

parser = argparse.ArgumentParser(description="Generate speech from a text file in data/.")
parser.add_argument(
    "--text-file",
    required=True,
    help="Text filename under data/ (e.g., togaf.txt) or a direct path.",
)
args = parser.parse_args()

text_path = Path(args.text_file)
if not text_path.is_absolute():
    if text_path.parts and text_path.parts[0] == "data":
        text_path = text_path
    else:
        text_path = Path("data") / text_path

if not text_path.exists():
    raise FileNotFoundError(f"Text file not found: {text_path}")

text_bytes = text_path.read_bytes()
text = text_bytes.decode("utf-8").strip()
if not text:
    raise ValueError(f"Text file is empty: {text_path}")

short_hash = hashlib.sha256(text_bytes).hexdigest()[:8]
output_audio = text_path.with_name(f"{text_path.stem}_{short_hash}.wav")

print("Loading VITS text-to-speech model...")
# Ensure local model exists to avoid unexpected network calls.
model_dir = Path("model_tts")
if not model_dir.exists():
    raise FileNotFoundError(
        "Local model not found at 'model_tts'. Run:\n"
        "  python download_hf_minimal.py --repo-id facebook/mms-tts-eng --output-dir model_tts\n"
        "Then rerun generate.py."
    )
# Load model and tokenizer from local folder
model = VitsModel.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Generating speech on {device}...")
# Split long text into smaller chunks to avoid OOM kills.
sentences = re.split(r"(?<=[.!?])\s+", text)
chunks = []
current = []
current_len = 0
for sentence in sentences:
    sentence = sentence.strip()
    if not sentence:
        continue
    if current_len + len(sentence) + 1 > MAX_CHARS_PER_CHUNK and current:
        chunks.append(" ".join(current))
        current = [sentence]
        current_len = len(sentence)
    else:
        current.append(sentence)
        current_len += len(sentence) + 1
if current:
    chunks.append(" ".join(current))

audio_chunks = []
for idx, chunk in enumerate(chunks, start=1):
    print(f"  Chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
    inputs = tokenizer(chunk, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_chunks.append(model(**inputs).waveform)

# Save audio file
import scipy.io.wavfile as wavfile
import numpy as np

# Convert to numpy and save
audio_data = torch.cat(audio_chunks, dim=-1).squeeze().cpu().numpy()
sample_rate = model.config.sampling_rate

# Normalize audio to int16 range
audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

wavfile.write(str(output_audio), rate=sample_rate, data=audio_data)

print(f"Audio saved → {output_audio}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
