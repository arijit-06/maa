import os
import io
import shutil
import re
import sys
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(__file__).resolve().parent

# File extensions considered binary (copy as-is)
BINARY_EXT = {'.h5', '.tflite', '.png', '.jpg', '.jpeg', '.gif', '.bin'}

# Text extensions to copy and strip comments
TEXT_EXT = {'.py', '.h', '.c', '.cpp', '.hpp', '.json', '.csv', '.txt'}


def strip_python_comments(src_text):
    """Remove Python comments and docstrings while keeping code intact using tokenize."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src_text).readline)
    except Exception:
        return src_text

    out = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    for tok_type, tok_string, start, end, line in tokens:
        if tok_type == tokenize.COMMENT:
            continue
        if tok_type == tokenize.STRING:
            # Skip module-level docstrings: when the string occurs as first token
            if prev_toktype == tokenize.INDENT or prev_toktype == tokenize.NEWLINE or last_col == 0:
                # Heuristic: treat as docstring if at start of module, class, or function
                # We'll skip it entirely
                prev_toktype = tok_type
                last_col = end[1]
                continue
        out.append(tok_string)
        prev_toktype = tok_type
        last_col = end[1]

    return "".join(out)


def strip_c_comments(text):
    # Remove /* */ block comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove // line comments
    text = re.sub(r'//.*', '', text)
    return text


def should_copy_binary(ext):
    return ext.lower() in BINARY_EXT


def should_strip_text(ext):
    return ext.lower() in TEXT_EXT


def process_file(src_path: Path, dst_path: Path):
    ext = src_path.suffix.lower()

    if should_copy_binary(ext):
        shutil.copy2(src_path, dst_path)
        return 'binary_copied'

    if should_strip_text(ext):
        # Read as text
        try:
            text = src_path.read_text(encoding='utf-8')
        except Exception:
            # Fallback: binary copy
            shutil.copy2(src_path, dst_path)
            return 'binary_fallback'

        if ext == '.py':
            cleaned = strip_python_comments(text)
        elif ext in {'.c', '.h', '.cpp', '.hpp'}:
            cleaned = strip_c_comments(text)
        else:
            # For JSON/CSV/TXT: no standard comments; copy as-is
            cleaned = text

        dst_path.write_text(cleaned, encoding='utf-8')
        return 'stripped'

    # Default: binary copy
    shutil.copy2(src_path, dst_path)
    return 'binary_default'


def main():
    src_files = [p for p in ROOT.iterdir() if p.name != DEPLOY.name]
    results = []

    for p in src_files:
        if p.is_dir():
            # Skip directories for now
            continue

        dst = DEPLOY / p.name
        status = process_file(p, dst)
        results.append((p.name, status))
        print(f"{p.name} -> {dst}  [{status}]")

    print("\nDone. Files written into deploy/ (comment-stripped where applicable).")


if __name__ == '__main__':
    main()
