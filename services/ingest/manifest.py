from __future__ import annotations

import json
from pathlib import Path

from .models import FrameEntry


def build_manifest(frame_dir: Path) -> tuple[Path, list[FrameEntry]]:
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))
    entries: list[FrameEntry] = []

    for i, frame in enumerate(frame_files):
        entries.append(
            FrameEntry(
                frame_index=i,
                timestamp_seconds=float(i),
                uri=str(frame),
            )
        )

    manifest_path = frame_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([entry.model_dump() for entry in entries], indent=2), encoding="utf-8"
    )
    return manifest_path, entries
