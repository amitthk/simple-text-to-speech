"""
Download a minimal set of files for a Hugging Face model to a local folder.

Example:
  python download_hf_minimal.py --repo-id facebook/mms-tts-eng --output-dir model_tts
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

# ------------------ CONFIG ------------------
DEFAULT_INCLUDE = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "*.model",
    "*.safetensors",
    "pytorch_model.bin",
    "model.safetensors",
]
# -------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download minimal files for a Hugging Face model."
    )
    parser.add_argument("--repo-id", required=True, help="Model repo, e.g. facebook/mms-tts-eng")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to store the model files.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=DEFAULT_INCLUDE,
        help="Optional allow-list of file patterns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.include,
    )

    print(f"Downloaded minimal files to {output_dir}")


if __name__ == "__main__":
    main()
