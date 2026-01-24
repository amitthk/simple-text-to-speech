#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

INPUT_PATH = Path("data/Transcript_TOGAF_EA_Foundation.txt")
OUTPUT_DIR = Path("data/togaf_modules")

MODULE_HEADER_RE = re.compile(r"^MODULE\s*\d+\s*:", re.MULTILINE)

DROP_LINE_PATTERNS = [
    re.compile(r"^Graphical user interface.*", re.IGNORECASE),
    re.compile(r"^Description automatically generated.*", re.IGNORECASE),
    re.compile(r"^Diagram.*generated.*", re.IGNORECASE),
    re.compile(r"^S\d*\s*Figure\b.*", re.IGNORECASE),
    re.compile(r"^Figure\b.*", re.IGNORECASE),
    re.compile(r"^Table\b.*", re.IGNORECASE),
]


def _is_all_caps_line(value: str) -> bool:
    letters = [c for c in value if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _is_heading_line(value: str) -> bool:
    return _is_all_caps_line(value) or value.startswith("Definition") or value.startswith("DEFINITION")


def _drop_line(value: str) -> bool:
    return any(pattern.match(value) for pattern in DROP_LINE_PATTERNS)


def _normalize_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = re.sub(r"MODULE\s*(\d+)\s*:", r"MODULE \1:", text, flags=re.MULTILINE)
    start_match = re.search(r"(?m)^MODULE\s*0\s*:", text)
    if start_match:
        text = text[start_match.start() :]
    return text


def _split_modules(text: str) -> list[str]:
    matches = list(MODULE_HEADER_RE.finditer(text))
    if not matches:
        raise ValueError("No module headers found in transcript.")
    modules = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        modules.append(text[start:end].strip())
    return modules


def _merge_module_heading(lines: list[str]) -> list[str]:
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if MODULE_HEADER_RE.match(line):
            parts = [line]
            i += 1
            while i < len(lines) and lines[i]:
                candidate = lines[i]
                if _is_all_caps_line(candidate) or candidate in {":", "-", "--"}:
                    parts.append(candidate)
                    i += 1
                    continue
                break
            heading = re.sub(r"\s+", " ", " ".join(parts)).strip().upper()
            merged.append(heading)
            continue
        if _is_all_caps_line(line):
            merged.append(line.upper())
        else:
            merged.append(line)
        i += 1
    return merged


def _merge_all_caps_blocks(lines: list[str]) -> list[str]:
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line and _is_all_caps_line(line) and not MODULE_HEADER_RE.match(line):
            parts = [line]
            i += 1
            while i < len(lines) and lines[i] and _is_all_caps_line(lines[i]):
                fragment = lines[i]
                if re.match(r"^[A-Z][?!.,]?$", fragment):
                    parts[-1] = f"{parts[-1]}{fragment}"
                else:
                    parts.append(fragment)
                i += 1
            merged.append(" ".join(parts))
            continue
        merged.append(line)
        i += 1
    return merged


def _reflow_lines(lines: list[str]) -> list[str]:
    output = []
    current = ""
    glue_next = False

    def flush() -> None:
        nonlocal current
        if current:
            output.append(re.sub(r"\s+", " ", current).strip())
            current = ""

    for line in lines:
        if not line:
            flush()
            output.append("")
            glue_next = False
            continue
        if _is_heading_line(line):
            flush()
            output.append(line)
            glue_next = False
            continue
        line = re.sub(r"\s+", " ", line)
        if glue_next:
            current += line
        else:
            current = f"{current} {line}".strip() if current else line
        if line.endswith("-"):
            current = current[:-1]
            glue_next = True
        else:
            glue_next = False
    flush()

    cleaned = []
    prev_blank = True
    for line in output:
        if line == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    if cleaned and cleaned[0] == "":
        cleaned = cleaned[1:]
    if cleaned and cleaned[-1] == "":
        cleaned = cleaned[:-1]
    return cleaned


def _clean_lines(chunk: str) -> list[str]:
    cleaned = []
    for line in (line.strip() for line in chunk.splitlines()):
        if not line:
            cleaned.append("")
            continue
        if _drop_line(line):
            continue
        cleaned.append(line)
    cleaned.append("")
    return cleaned


def _module_number(chunk: str) -> int:
    match = re.search(r"MODULE\s*(\d+)\s*:", chunk)
    if not match:
        raise ValueError("Module number not found.")
    return int(match.group(1))


def main() -> None:
    raw = INPUT_PATH.read_text(encoding="utf-8")
    text = _normalize_text(raw)
    modules = _split_modules(text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for chunk in modules:
        number = _module_number(chunk)
        lines = _clean_lines(chunk)
        lines = _merge_module_heading(lines)
        lines = _merge_all_caps_blocks(lines)
        lines = _reflow_lines(lines)
        out_path = OUTPUT_DIR / f"module_{number:02d}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
