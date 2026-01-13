# Simple Text-to-Speech

Generate WAV audio from a text file using the offline MMS-TTS (VITS) model.

## Requirements
- Python 3.9+ (macOS or Linux)
- `pip` for dependency installation

## Install (macOS or Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download the Model (Required)
This project expects the English MMS-TTS model at `model_tts/`.
```bash
python download_hf_minimal.py --repo-id facebook/mms-tts-eng --output-dir model_tts
```

## Run
Put your text files in `data/` and pass the filename:
```bash
python generate.py --text-file notes.txt
```

The output WAV is written next to the input file with a short content hash:
`data/notes_<hash>.wav`.
