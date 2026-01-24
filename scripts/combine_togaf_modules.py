#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine TOGAF module files into a single text file.")
    parser.add_argument(
        "--modules-dir",
        default="data/togaf_modules",
        help="Directory containing module_XX.txt files.",
    )
    parser.add_argument(
        "--output",
        default="data/togaf.txt",
        help="Output text file path.",
    )
    parser.add_argument(
        "--pattern",
        default="module_*.txt",
        help="Glob pattern for module files.",
    )
    args = parser.parse_args()

    modules_dir = Path(args.modules_dir)
    output_path = Path(args.output)
    module_paths = sorted(modules_dir.glob(args.pattern))
    if not module_paths:
        raise FileNotFoundError(f"No module files found in {modules_dir} matching {args.pattern}")

    combined_parts = []
    for path in module_paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            combined_parts.append(text)

    combined_text = "\n\n".join(combined_parts) + "\n"
    output_path.write_text(combined_text, encoding="utf-8")
    print(f"Wrote {output_path} from {len(module_paths)} module files.")


if __name__ == "__main__":
    main()
