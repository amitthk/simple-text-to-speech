#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine cached chunk WAVs into a single file.")
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Path to the chunk cache directory (e.g., audio_cache/togaf_<hash>).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        help="Silence duration to insert between chunks.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    chunk_paths = sorted(cache_dir.glob("chunk_*.wav"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk WAVs found in {cache_dir}")

    sample_rate = None
    combined = []
    for chunk_path in chunk_paths:
        rate, data = wavfile.read(str(chunk_path))
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Sample rate mismatch in {chunk_path}: {rate} != {sample_rate}")
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        combined.append(data)

    silence = np.zeros(int(sample_rate * args.pause_seconds), dtype=np.float32)
    audio_data = []
    for idx, data in enumerate(combined):
        audio_data.append(data)
        if idx < len(combined) - 1:
            audio_data.append(silence)

    output = np.concatenate(audio_data) if audio_data else np.array([], dtype=np.float32)
    if output.size == 0:
        raise ValueError("No audio data to write.")

    output_int16 = np.int16(output / np.max(np.abs(output)) * 32767)
    output_path = Path(args.output)
    wavfile.write(str(output_path), rate=sample_rate, data=output_int16)
    print(f"Wrote {output_path} from {len(chunk_paths)} chunks.")


if __name__ == "__main__":
    main()
