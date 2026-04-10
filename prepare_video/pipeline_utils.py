from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path


SOURCE_DIR_MAP = {
    "右足": "right_leg",
    "左足": "left_leg",
    "固定無し": "nothing",
}

VIDEO_VIEW_FILES = {
    "above": "above.mp4",
    "front": "front.mp4",
    "side": "side.mp4",
}

RAW_LABEL_TO_CLASS = {
    "nothing": "normal",
    "right_leg": "weight",
    "left_leg": "weight",
}

SEGMENT_FILE_PATTERN = re.compile(
    r"^(?P<session>.+)_(?P<view>front|side|above)_(?P<segment>\d{4})$"
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


def parse_segment_video_path(path: Path) -> SegmentVideoFile | None:
    match = SEGMENT_FILE_PATTERN.match(path.stem)
    if match is None or len(path.parts) < 3:
        return None

    source_label = path.parent.name
    target_label = RAW_LABEL_TO_CLASS.get(source_label, source_label)

    return SegmentVideoFile(
        path=path,
        group_id=path.parent.parent.name,
        source_label=source_label,
        target_label=target_label,
        session_id=match.group("session"),
        view=match.group("view"),
        segment_index=int(match.group("segment")),
    )
