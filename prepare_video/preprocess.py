from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline_utils import ensure_dir, normalize_source_label


RAW_VIDEO_FILES = {
    "front": "full_ap.mp4",
    "side": "full_lat.mp4",
}
DATE_DIR_PATTERN = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class PreprocessConfig:
    input_root: Path
    output_root: Path
    model_path: Path
    monitoring_frames: int = 10
    stable_walk_frames: int = 10
    min_segment_seconds: float = 2.0
    detection_confidence: float = 0.25
    min_bbox_area_ratio: float = 0.005
    target_size: int = 512


@dataclass(frozen=True)
class VideoProperties:
    frame_count: int
    fps: float


@dataclass(frozen=True)
class SessionContext:
    date_code: str
    source_label: str
    session_id: str
    session_dir: Path
    video_paths: dict[str, Path]
    frame_count: int
    fps: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and crop synchronized full AP/LAT walk videos into paired segments."
    )
    parser.add_argument("--input-root", default="/workspace/data/raw")
    parser.add_argument("--output-root", default="/workspace/data/segment")
    parser.add_argument("--model-path", default="/workspace/project/yolo12n.pt")
    parser.add_argument("--monitoring-frames", type=int, default=10)
    parser.add_argument("--stable-walk-frames", type=int, default=10)
    parser.add_argument("--min-segment-seconds", type=float, default=2.0)
    parser.add_argument("--detection-confidence", type=float, default=0.25)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.005)
    parser.add_argument("--target-size", type=int, default=512)
    return parser


def parse_args(argv: list[str] | None = None) -> PreprocessConfig:
    args = build_parser().parse_args(argv)
    return PreprocessConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        model_path=Path(args.model_path),
        monitoring_frames=args.monitoring_frames,
        stable_walk_frames=args.stable_walk_frames,
        min_segment_seconds=args.min_segment_seconds,
        detection_confidence=args.detection_confidence,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        target_size=args.target_size,
    )


def load_model(model_path: Path):
    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for processing.")
    return model


def get_video_properties(video_path: Path) -> VideoProperties | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    properties = VideoProperties(
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
    )
    cap.release()

    if properties.frame_count <= 0 or properties.fps <= 0:
        return None
    return properties


def find_date_code(relative_parts: tuple[str, ...]) -> str | None:
    for part in reversed(relative_parts):
        if DATE_DIR_PATTERN.fullmatch(part):
            return part
    return None


def build_session_id(relative_parts: tuple[str, ...]) -> str:
    safe_parts = [part.replace(" ", "_").replace("/", "_") for part in relative_parts]
    return "_".join(safe_parts)


def infer_source_label(relative_parts: tuple[str, ...], date_index: int) -> str | None:
    label_candidates = relative_parts[:date_index]
    if not label_candidates:
        return None

    known_labels = {"ASD", "ASD_not", "DHS", "LCS", "HipOA"}
    normalized_candidates = [normalize_source_label(part) for part in label_candidates]
    for candidate in reversed(normalized_candidates):
        if candidate in known_labels:
            return candidate

    return normalized_candidates[-1]


def build_session_context(input_root: Path, session_dir: Path) -> SessionContext | None:
    relative_parts = session_dir.relative_to(input_root).parts
    date_code = find_date_code(relative_parts)
    if date_code is None:
        print(f"Skipping {session_dir}: no date directory like YYYYMMDD was found in its path.")
        return None

    date_index = relative_parts.index(date_code)
    source_label = infer_source_label(relative_parts, date_index)
    if source_label is None:
        print(f"Skipping {session_dir}: could not infer source label from its path.")
        return None

    session_id = build_session_id(relative_parts)

    video_paths = {
        view: session_dir / file_name
        for view, file_name in RAW_VIDEO_FILES.items()
    }
    missing_files = [str(path) for path in video_paths.values() if not path.exists()]
    if missing_files:
        print(f"Skipping {session_dir}: missing files {missing_files}")
        return None

    properties = {
        view: get_video_properties(path)
        for view, path in video_paths.items()
    }
    if any(prop is None for prop in properties.values()):
        print(f"Skipping {session_dir}: failed to read video metadata.")
        return None

    frame_counts = {prop.frame_count for prop in properties.values()}
    fps_values = {round(prop.fps, 3) for prop in properties.values()}
    if len(frame_counts) != 1 or len(fps_values) != 1:
        print(
            f"Skipping {session_dir}: frame counts or FPS are inconsistent "
            f"({frame_counts=}, {fps_values=})."
        )
        return None

    reference = next(iter(properties.values()))
    return SessionContext(
        date_code=date_code,
        source_label=source_label,
        session_id=session_id,
        session_dir=session_dir,
        video_paths=video_paths,
        frame_count=reference.frame_count,
        fps=reference.fps,
    )


def iter_session_dirs(input_root: Path) -> list[Path]:
    required_files = set(RAW_VIDEO_FILES.values())
    session_dirs: list[Path] = []

    for candidate in sorted(input_root.rglob("*")):
        if not candidate.is_dir():
            continue

        present_files = {
            child.name
            for child in candidate.iterdir()
            if child.is_file()
        }
        if required_files.issubset(present_files):
            session_dirs.append(candidate)

    return session_dirs


def select_detection_box(
    results,
    frame_shape: tuple[int, int, int],
    confidence: float,
    min_bbox_area_ratio: float,
    strategy: str,
) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    min_area = height * width * min_bbox_area_ratio
    candidates: list[tuple[int, int, int, int]] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            if int(box.cls) != 0 or float(box.conf) < confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            box_height = y2 - y1
            box_width = x2 - x1
            if box_width <= 0 or box_height <= 0:
                continue
            if box_width > box_height:
                continue
            if box_width * box_height < min_area:
                continue

            candidates.append((x1, y1, x2, y2))

    if not candidates:
        return None

    if strategy == "leftmost":
        return min(candidates, key=lambda candidate: candidate[0])
    if strategy == "rightmost":
        return max(candidates, key=lambda candidate: candidate[0])
    raise ValueError(f"Unsupported detection strategy: {strategy}")


def render_person_frame(frame, bbox, target_size: int) -> Any:
    import cv2
    import numpy as np

    output_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    if bbox is None:
        return output_frame

    x1, y1, x2, y2 = map(int, bbox)
    height, width = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x1 >= x2 or y1 >= y2:
        return output_frame

    person_crop = frame[y1:y2, x1:x2]
    person_height, person_width = person_crop.shape[:2]
    if person_height <= 0 or person_width <= 0:
        return output_frame

    scale = target_size / person_height
    resized_width = max(1, int(round(person_width * scale)))
    resized = cv2.resize(
        person_crop,
        (resized_width, target_size),
        interpolation=cv2.INTER_LINEAR,
    )

    start_x = (target_size - resized_width) // 2
    end_x = start_x + resized_width

    image_start_x = 0
    image_end_x = resized_width
    paste_start_x = start_x
    paste_end_x = end_x

    if paste_start_x < 0:
        image_start_x = -paste_start_x
        paste_start_x = 0
    if paste_end_x > target_size:
        image_end_x = resized_width - (paste_end_x - target_size)
        paste_end_x = target_size

    if paste_start_x < paste_end_x:
        output_frame[:, paste_start_x:paste_end_x] = resized[:, image_start_x:image_end_x]
    return output_frame


def infer_walk_direction(center_x_history: list[float], movement_threshold: float = 5.0) -> str:
    if len(center_x_history) < 2:
        return "Unknown"

    first_x = center_x_history[0]
    last_x = center_x_history[-1]
    if last_x < first_x - movement_threshold:
        return "Left"
    if last_x > first_x + movement_threshold:
        return "Right"
    return "Standing"


def write_segment_video(output_path: Path, frames: list[Any], fps: float, target_size: int) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_size, target_size),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def flush_segment(
    context: SessionContext,
    config: PreprocessConfig,
    segment_index: int,
    front_frames: list[Any],
    side_frames: list[Any],
) -> int:
    min_segment_frames = max(1, int(round(context.fps * config.min_segment_seconds)))
    if len(front_frames) < min_segment_frames or len(side_frames) < min_segment_frames:
        return 0

    output_dir = ensure_dir(config.output_root / context.date_code / context.source_label)
    front_output_path = output_dir / f"{context.session_id}_front_{segment_index:04d}.mp4"
    side_output_path = output_dir / f"{context.session_id}_side_{segment_index:04d}.mp4"

    write_segment_video(front_output_path, front_frames, context.fps, config.target_size)
    write_segment_video(side_output_path, side_frames, context.fps, config.target_size)
    return 1


def preprocess_session(context: SessionContext, model, config: PreprocessConfig) -> int:
    import cv2

    front_cap = cv2.VideoCapture(str(context.video_paths["front"]))
    side_cap = cv2.VideoCapture(str(context.video_paths["side"]))
    if not front_cap.isOpened() or not side_cap.isOpened():
        print(f"Skipping {context.session_dir}: could not open synchronized input videos.")
        front_cap.release()
        side_cap.release()
        return 0

    center_x_history: list[float] = []
    previous_direction = "Unknown"
    current_stable_direction = "Unknown"
    stable_direction_count = 0
    recording = False
    segment_index = 0
    created_segments = 0

    buffered_front_frames: list[Any] = []
    buffered_side_frames: list[Any] = []

    while front_cap.isOpened() and side_cap.isOpened():
        front_ok, front_frame = front_cap.read()
        side_ok, side_frame = side_cap.read()
        if not front_ok or not side_ok:
            break

        front_results = model(front_frame, classes=0, conf=config.detection_confidence, verbose=False)
        side_results = model(side_frame, classes=0, conf=config.detection_confidence, verbose=False)

        front_bbox = select_detection_box(
            front_results,
            front_frame.shape,
            confidence=config.detection_confidence,
            min_bbox_area_ratio=config.min_bbox_area_ratio,
            strategy="leftmost",
        )
        side_bbox = select_detection_box(
            side_results,
            side_frame.shape,
            confidence=config.detection_confidence,
            min_bbox_area_ratio=config.min_bbox_area_ratio,
            strategy="rightmost",
        )

        front_detected = front_bbox is not None
        side_detected = side_bbox is not None

        if side_detected:
            side_center_x = (side_bbox[0] + side_bbox[2]) / 2
            center_x_history.append(side_center_x)
            if len(center_x_history) > config.monitoring_frames:
                center_x_history.pop(0)
            side_direction = infer_walk_direction(center_x_history)
        else:
            center_x_history.clear()
            side_direction = "Unknown"

        if side_direction == previous_direction:
            stable_direction_count += 1
        else:
            stable_direction_count = 0

        if (
            stable_direction_count >= config.stable_walk_frames
            and side_direction in {"Left", "Right"}
            and not recording
        ):
            recording = True
            current_stable_direction = side_direction
            buffered_front_frames = []
            buffered_side_frames = []

        if recording:
            end_of_section = (
                side_direction != current_stable_direction
                or not front_detected
                or not side_detected
            )

            if not end_of_section:
                buffered_front_frames.append(
                    render_person_frame(front_frame, front_bbox, config.target_size)
                )
                buffered_side_frames.append(
                    render_person_frame(side_frame, side_bbox, config.target_size)
                )
            else:
                created_segments += flush_segment(
                    context=context,
                    config=config,
                    segment_index=segment_index,
                    front_frames=buffered_front_frames,
                    side_frames=buffered_side_frames,
                )
                if buffered_front_frames and buffered_side_frames:
                    segment_index += 1

                buffered_front_frames = []
                buffered_side_frames = []
                recording = False
                current_stable_direction = "Unknown"

        previous_direction = side_direction if side_detected else "Unknown"

    if recording and buffered_front_frames and buffered_side_frames:
        created_segments += flush_segment(
            context=context,
            config=config,
            segment_index=segment_index,
            front_frames=buffered_front_frames,
            side_frames=buffered_side_frames,
        )

    front_cap.release()
    side_cap.release()
    return created_segments


def preprocess_dataset(config: PreprocessConfig) -> None:
    if not config.input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_root}")

    ensure_dir(config.output_root)
    model = load_model(config.model_path)

    total_segments = 0
    processed_sessions = 0

    session_dirs = iter_session_dirs(config.input_root)
    if not session_dirs:
        raise RuntimeError(
            "No raw sessions were found. Expected directories containing "
            "`full_ap.mp4` and `full_lat.mp4`."
        )

    for session_dir in session_dirs:
        context = build_session_context(config.input_root, session_dir)
        if context is None:
            continue

        print(f"\nProcessing {context.session_id}")
        processed_sessions += 1
        total_segments += preprocess_session(context, model, config)

    print(
        f"\nFinished preprocessing. processed_sessions={processed_sessions}, "
        f"created_segments={total_segments}"
    )


if __name__ == "__main__":
    preprocess_dataset(parse_args())
