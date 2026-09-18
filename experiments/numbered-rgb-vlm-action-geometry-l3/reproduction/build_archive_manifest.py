#!/usr/bin/env python3
"""Build a deterministic integrity manifest for this archived experiment."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit" / "archive_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int | str:
    if path.suffix != ".jsonl":
        return ""
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256", "jsonl_rows"])
        for path in paths:
            writer.writerow(
                [
                    path.relative_to(ROOT).as_posix(),
                    path.stat().st_size,
                    sha256(path),
                    line_count(path),
                ]
            )


if __name__ == "__main__":
    main()
