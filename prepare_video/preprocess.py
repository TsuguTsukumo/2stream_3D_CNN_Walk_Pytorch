from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline_utils import SOURCE_DIR_MAP, VIDEO_VIEW_FILES, ensure_dir


@dataclass(frozen=True)
class PreprocessConfig:
    input_root: Path
    output_root: Path
    model_path: Path
    max_allowed_gap: int = 5
    min_segment_frames: int = 30
    detection_confidence: float = 0.25
    target_size: int = 512


@dataclass(frozen=True)
class VideoProperties:
    frame_count: int
    fps: float


@dataclass(frozen=True)
class SessionContext:
    date_code: str
    source_label: str
    output_label: str
    video_paths: dict[str, Path]
    frame_count: int
    fps: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop and segment synchronized walk videos."
    )
    parser.add_argument("--input-root", default="/workspace/data/raw")
    parser.add_argument("--output-root", default="/workspace/data/segment")
    parser.add_argument("--model-path", default="/workspace/project/yolo12n.pt")
    parser.add_argument("--max-allowed-gap", type=int, default=5)
    parser.add_argument("--min-segment-frames", type=int, default=30)
    parser.add_argument("--detection-confidence", type=float, default=0.25)
    parser.add_argument("--target-size", type=int, default=512)
    return parser


def parse_args(argv: list[str] | None = None) -> PreprocessConfig:
    args = build_parser().parse_args(argv)
    return PreprocessConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        model_path=Path(args.model_path),
        max_allowed_gap=args.max_allowed_gap,
        min_segment_frames=args.min_segment_frames,
        detection_confidence=args.detection_confidence,
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


def build_session_context(
    date_path: Path,
    source_label: str,
    output_label: str,
) -> SessionContext | None:
    session_dir = date_path / source_label
    if not session_dir.is_dir():
        return None

    video_paths = {
        view: session_dir / file_name
        for view, file_name in VIDEO_VIEW_FILES.items()
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
        date_code=date_path.name,
        source_label=source_label,
        output_label=output_label,
        video_paths=video_paths,
        frame_count=reference.frame_count,
        fps=reference.fps,
    )


def detect_people(
    video_path: Path,
    model,
    confidence: float,
) -> dict[int, Any]:
    import cv2
    from tqdm import tqdm

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    detections: dict[int, Any] = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=total_frames, desc=f"Detecting {video_path.name}")

    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, classes=0, conf=confidence, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            rightmost_box = max(boxes, key=lambda box: box[0])
            detections[frame_index] = rightmost_box.astype(int)

        frame_index += 1
        progress.update(1)

    progress.close()
    cap.release()
    return detections


def find_continuous_segments(
    indices: list[int],
    min_length: int,
    max_gap: int,
) -> list[list[int]]:
    if not indices:
        return []

    segments: list[list[int]] = []
    current_segment = [indices[0]]

    for index in range(1, len(indices)):
        gap = indices[index] - indices[index - 1] - 1
        if gap <= max_gap:
            current_segment.append(indices[index])
            continue

        if len(current_segment) >= min_length:
            segments.append(current_segment)
        current_segment = [indices[index]]

    if len(current_segment) >= min_length:
        segments.append(current_segment)

    return segments


def render_person_frame(
    frame,
    bbox,
    target_size: int,
) -> Any:
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


def write_segment_video(
    video_path: Path,
    output_path: Path,
    frame_indices: list[int],
    detections: dict[int, Any],
    fps: float,
    target_size: int,
) -> None:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_size, target_size),
    )

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            writer.write(np.zeros((target_size, target_size, 3), dtype=np.uint8))
            continue
        writer.write(
            render_person_frame(frame, detections.get(frame_index), target_size)
        )

    writer.release()
    cap.release()


def preprocess_session(
    context: SessionContext,
    model,
    config: PreprocessConfig,
) -> int:
    from tqdm import tqdm

    detections = {
        view: detect_people(path, model, config.detection_confidence)
        for view, path in context.video_paths.items()
    }

    shared_indices = sorted(
        set(detections["front"].keys()) & set(detections["side"].keys())
    )
    segments = find_continuous_segments(
        shared_indices,
        min_length=config.min_segment_frames,
        max_gap=config.max_allowed_gap,
    )
    if not segments:
        print(f"No synchronized segments found for {context.date_code}/{context.output_label}.")
        return 0

    output_dir = ensure_dir(config.output_root / context.date_code / context.output_label)
    progress = tqdm(
        total=len(segments),
        desc=f"Writing {context.date_code}/{context.output_label}",
    )

    for segment_index, frame_indices in enumerate(segments):
        for view, video_path in context.video_paths.items():
            output_path = output_dir / f"{context.date_code}_{view}_{segment_index:04d}.mp4"
            write_segment_video(
                video_path=video_path,
                output_path=output_path,
                frame_indices=frame_indices,
                detections=detections[view],
                fps=context.fps,
                target_size=config.target_size,
            )
        progress.update(1)

    progress.close()
    return len(segments)


def preprocess_dataset(config: PreprocessConfig) -> None:
    if not config.input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_root}")

    ensure_dir(config.output_root)
    model = load_model(config.model_path)

    total_segments = 0
    processed_sessions = 0

    for date_path in sorted(config.input_root.iterdir()):
        if not date_path.is_dir() or not date_path.name[:8].isdigit():
            continue

        print(f"\nProcessing {date_path.name}")
        for source_label, output_label in SOURCE_DIR_MAP.items():
            context = build_session_context(date_path, source_label, output_label)
            if context is None:
                continue

            processed_sessions += 1
            total_segments += preprocess_session(context, model, config)

    print(
        f"\nFinished preprocessing. processed_sessions={processed_sessions}, "
        f"created_segments={total_segments}"
    )


if __name__ == "__main__":
    preprocess_dataset(parse_args())
