# Simple Text-to-Speech

Generate WAV audio from a text file using an offline VITS text-to-speech model.

## Requirements
- Python 3.9+ (macOS or Linux)
- `pip` for dependency installation
- System TTS dependency for phonemizer: `espeak-ng` (macOS/Linux)

## Install (macOS or Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install phonemizer's system dependency:
```bash
brew install espeak-ng
```

## Download the Model (Required)
This project uses the female LJSpeech VITS model at `model_tts_female/`.
```bash
python download_hf_minimal.py --repo-id kakao-enterprise/vits-ljs --output-dir model_tts_female
```

## Run
Put your text files in `data/` and pass the filename:
```bash
python generate.py --text-file notes.txt
```

The output WAV is written next to the input file with a short content hash:
`data/notes_<hash>.wav`.

## Dependencies
- Python: `torch`, `transformers`, `scipy`, `numpy`, `huggingface_hub`, `phonemizer`
- System: `espeak-ng` (required by `phonemizer`)
