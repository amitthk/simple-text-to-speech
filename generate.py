"""
Text-to-Speech using MMS-TTS (Offline Setup)

INSTALLATION:
1. Install dependencies:
   pip install transformers torch scipy numpy huggingface_hub

2. Download model to local folder:
   python download_hf_minimal.py --repo-id kakao-enterprise/vits-ljs --output-dir model_tts_female

3. Run script (will use local model):
   python generate.py --text-file togaf.txt

Model size: ~400MB
Location: model_tts_female/ folder
"""

import argparse
import hashlib
import re
from pathlib import Path

import torch
from transformers import VitsModel, AutoTokenizer
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal

# ------------------ CONFIG ------------------
LANGUAGE = "en"  # Language code
SPEAKER_ID = 0  # Single-speaker model; kept for compatibility
MAX_CHARS_PER_CHUNK = 600  # Reduce memory use for long inputs
PAUSE_SECONDS = 2.0  # Pause between paragraphs/sections
SPEED_FACTOR = 1.0  # >1.0 is faster, <1.0 is slower
MODEL_DIR = Path("model_tts_female")
CACHE_DIR = Path("audio_cache")
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
if not MODEL_DIR.exists():
    raise FileNotFoundError(
        "Local model not found at 'model_tts_female'. Run:\n"
        "  python download_hf_minimal.py --repo-id kakao-enterprise/vits-ljs --output-dir model_tts_female\n"
        "Then rerun generate.py."
    )
# Load model and tokenizer from local folder
model = VitsModel.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Generating speech on {device}...")

def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()

def _is_all_caps_line(value: str) -> bool:
    letters = [c for c in value if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)

def _is_definition_line(value: str) -> bool:
    return value.startswith("Definition") or value.startswith("DEFINITION")

def _split_paragraphs(value: str) -> list[str]:
    paragraphs = []
    current = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(_normalize_whitespace(" ".join(current)))
                current = []
            continue
        if _is_all_caps_line(line) or _is_definition_line(line):
            if current:
                paragraphs.append(_normalize_whitespace(" ".join(current)))
                current = []
            paragraphs.append(_normalize_whitespace(line))
            continue
        current.append(line)
    if current:
        paragraphs.append(_normalize_whitespace(" ".join(current)))
    return paragraphs

def _split_sentences(value: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", value) if s.strip()]

def _chunk_paragraph(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_CHARS_PER_CHUNK:
        return [paragraph]
    sentences = _split_sentences(paragraph)
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current_len + add_len > MAX_CHARS_PER_CHUNK and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += add_len
    if current:
        chunks.append(" ".join(current))
    return chunks

def _apply_speed(waveform: np.ndarray, speed_factor: float) -> np.ndarray:
    if speed_factor <= 0:
        raise ValueError("SPEED_FACTOR must be > 0")
    if speed_factor == 1.0:
        return waveform
    new_length = max(1, int(len(waveform) / speed_factor))
    return signal.resample(waveform, new_length).astype(np.float32)

paragraphs = _split_paragraphs(text)
chunk_plan = []
for idx, paragraph in enumerate(paragraphs):
    para_chunks = _chunk_paragraph(paragraph)
    for c_idx, chunk in enumerate(para_chunks):
        pause_after = (c_idx == len(para_chunks) - 1) and (idx < len(paragraphs) - 1)
        chunk_plan.append({"text": chunk, "pause_after": pause_after})

cache_dir = CACHE_DIR / f"{text_path.stem}_{short_hash}"
cache_dir.mkdir(parents=True, exist_ok=True)
chunk_paths = []

for idx, entry in enumerate(chunk_plan, start=1):
    chunk_text = entry["text"]
    chunk_path = cache_dir / f"chunk_{idx:04d}.wav"
    print(f"  Chunk {idx}/{len(chunk_plan)} ({len(chunk_text)} chars)")
    inputs = tokenizer(chunk_text, return_tensors="pt").to(device)
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze().cpu().numpy()
    waveform = _apply_speed(waveform, SPEED_FACTOR)
    chunk_paths.append((chunk_path, entry["pause_after"]))

    # Write chunk audio as float32 to preserve quality before final normalization.
    wavfile.write(str(chunk_path), rate=model.config.sampling_rate, data=waveform.astype(np.float32))

# Combine cached chunks with pauses.
sample_rate = model.config.sampling_rate
silence = np.zeros(int(sample_rate * PAUSE_SECONDS), dtype=np.float32)
combined = []

for chunk_path, pause_after in chunk_paths:
    rate, data = wavfile.read(str(chunk_path))
    if rate != sample_rate:
        raise ValueError(f"Sample rate mismatch in {chunk_path}: {rate} != {sample_rate}")
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    combined.append(data)
    if pause_after:
        combined.append(silence)

audio_data = np.concatenate(combined) if combined else np.array([], dtype=np.float32)
if audio_data.size == 0:
    raise ValueError("No audio data generated.")

# Normalize audio to int16 range
audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
wavfile.write(str(output_audio), rate=sample_rate, data=audio_data)

# Cleanup cached chunks
for chunk_path, _ in chunk_paths:
    chunk_path.unlink(missing_ok=True)
cache_dir.rmdir()

print(f"Audio saved → {output_audio}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
