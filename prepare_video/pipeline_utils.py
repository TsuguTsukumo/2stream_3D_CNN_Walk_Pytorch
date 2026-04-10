from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path


SEGMENT_FILE_PATTERN = re.compile(
    r"^(?P<session>.+)_(?P<view>front|side)_(?P<segment>\d{4})$"
)


@dataclass(frozen=True)
class SegmentVideoFile:
    path: Path
    group_id: str
    source_label: str
    target_label: str
    session_id: str
    view: str
    segment_index: int

    @property
    def pair_stem(self) -> str:
        return f"{self.session_id}_{self.source_label}_seg{self.segment_index:04d}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_source_label(label: str) -> str:
    normalized_key = label.strip().lower().replace("_", "-").replace(" ", "-")

    if normalized_key == "asd":
        return "ASD"
    if normalized_key in {"non-asd", "asd-not", "asdnot"}:
        return "ASD_not"
    if normalized_key == "dhs":
        return "DHS"
    if normalized_key == "lcs":
        return "LCS"
    if normalized_key == "hipoa":
        return "HipOA"

    return label.strip().replace(" ", "_")


def get_target_label(source_label: str) -> str:
    return "ASD" if normalize_source_label(source_label) == "ASD" else "ASD_not"


def parse_segment_video_path(path: Path) -> SegmentVideoFile | None:
    match = SEGMENT_FILE_PATTERN.match(path.stem)
    if match is None or len(path.parts) < 3:
        return None

    source_label = normalize_source_label(path.parent.name)
    target_label = get_target_label(source_label)

    return SegmentVideoFile(
        path=path,
        group_id=path.parent.parent.name,
        source_label=source_label,
        target_label=target_label,
        session_id=match.group("session"),
        view=match.group("view"),
        segment_index=int(match.group("segment")),
    )
