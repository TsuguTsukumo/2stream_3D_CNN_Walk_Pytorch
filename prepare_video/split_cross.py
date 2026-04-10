from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .pipeline_utils import parse_segment_video_path, reset_dir


@dataclass(frozen=True)
class SplitConfig:
    input_root: Path
    output_root: Path
    num_folds: int = 5
    random_state: int = 42
    clip_seconds: float = 1.0


@dataclass(frozen=True)
class SegmentPair:
    group_id: str
    source_label: str
    target_label: str
    session_id: str
    segment_index: int
    front_path: Path
    side_path: Path

    @property
    def output_stem(self) -> str:
        return f"{self.session_id}_{self.source_label}_seg{self.segment_index:04d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split synchronized AP/LAT segments into cross-validation folds."
    )
    parser.add_argument("--input-root", default="/workspace/data/segment")
    parser.add_argument("--output-root", default="/workspace/data/5folds")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--clip-seconds", type=float, default=1.0)
    return parser


def parse_args(argv: list[str] | None = None) -> SplitConfig:
    args = build_parser().parse_args(argv)
    return SplitConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        num_folds=args.num_folds,
        random_state=args.random_state,
        clip_seconds=args.clip_seconds,
    )


def collect_segment_pairs(input_root: Path) -> list[SegmentPair]:
    grouped_files: dict[tuple[str, str, str, int], dict[str, object]] = defaultdict(dict)
    skipped_files = 0

    for video_path in sorted(input_root.rglob("*.mp4")):
        info = parse_segment_video_path(video_path)
        if info is None or info.view not in {"front", "side"}:
            skipped_files += 1
            continue

        key = (
            info.group_id,
            info.source_label,
            info.session_id,
            info.segment_index,
        )
        grouped_files[key]["group_id"] = info.group_id
        grouped_files[key]["source_label"] = info.source_label
        grouped_files[key]["target_label"] = info.target_label
        grouped_files[key]["session_id"] = info.session_id
        grouped_files[key]["segment_index"] = info.segment_index
        grouped_files[key][info.view] = info.path

    segment_pairs: list[SegmentPair] = []
    incomplete_pairs = 0

    for grouped in grouped_files.values():
        front_path = grouped.get("front")
        side_path = grouped.get("side")
        if front_path is None or side_path is None:
            incomplete_pairs += 1
            continue

        segment_pairs.append(
            SegmentPair(
                group_id=str(grouped["group_id"]),
                source_label=str(grouped["source_label"]),
                target_label=str(grouped["target_label"]),
                session_id=str(grouped["session_id"]),
                segment_index=int(grouped["segment_index"]),
                front_path=front_path,
                side_path=side_path,
            )
        )

    print(
        f"Collected {len(segment_pairs)} synchronized AP/LAT segments "
        f"(skipped_files={skipped_files}, incomplete_pairs={incomplete_pairs})."
    )
    return segment_pairs


def split_pair_into_clips(
    pair: SegmentPair,
    ap_output_dir: Path,
    lat_output_dir: Path,
    clip_seconds: float,
) -> int:
    import cv2

    front_cap = cv2.VideoCapture(str(pair.front_path))
    side_cap = cv2.VideoCapture(str(pair.side_path))

    if not front_cap.isOpened() or not side_cap.isOpened():
        print(f"Skipping pair because it could not be opened: {pair.output_stem}")
        front_cap.release()
        side_cap.release()
        return 0

    fps_front = float(front_cap.get(cv2.CAP_PROP_FPS))
    fps_side = float(side_cap.get(cv2.CAP_PROP_FPS))
    if fps_front <= 0 or fps_side <= 0 or abs(fps_front - fps_side) > 0.001:
        print(f"Skipping pair because FPS does not match: {pair.output_stem}")
        front_cap.release()
        side_cap.release()
        return 0

    total_frames_front = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_side = int(side_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames_front, total_frames_side)
    frames_per_clip = max(int(round(fps_front * clip_seconds)), 1)

    if total_frames_front != total_frames_side:
        print(
            f"Pair frame count mismatch detected for {pair.output_stem}: "
            f"front={total_frames_front}, side={total_frames_side}. "
            f"Using the shorter stream."
        )

    if total_frames < frames_per_clip:
        front_cap.release()
        side_cap.release()
        return 0

    front_size = (
        int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    side_size = (
        int(side_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(side_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    clip_count = 0
    for start_frame in range(0, total_frames - frames_per_clip + 1, frames_per_clip):
        output_name = f"{pair.output_stem}_{clip_count:03d}.mp4"
        front_output_path = ap_output_dir / output_name
        side_output_path = lat_output_dir / output_name

        front_writer = cv2.VideoWriter(
            str(front_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_front,
            front_size,
        )
        side_writer = cv2.VideoWriter(
            str(side_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_front,
            side_size,
        )

        front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        side_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(frames_per_clip):
            front_ok, front_frame = front_cap.read()
            side_ok, side_frame = side_cap.read()
            if not front_ok or not side_ok:
                break
            front_writer.write(front_frame)
            side_writer.write(side_frame)

        front_writer.release()
        side_writer.release()
        clip_count += 1

    front_cap.release()
    side_cap.release()
    return clip_count


def prepare_dataset(config: SplitConfig) -> None:
    from sklearn.model_selection import KFold
    from tqdm import tqdm

    if not config.input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {config.input_root}")

    segment_pairs = collect_segment_pairs(config.input_root)
    if not segment_pairs:
        raise RuntimeError("No synchronized front/side segments were found.")

    group_ids = sorted({pair.group_id for pair in segment_pairs})
    if len(group_ids) < config.num_folds:
        raise ValueError(
            f"The number of groups ({len(group_ids)}) is smaller than "
            f"num_folds ({config.num_folds})."
        )

    reset_dir(config.output_root)

    k_fold = KFold(
        n_splits=config.num_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    total_clips = 0
    for fold_index, (train_indices, val_indices) in enumerate(k_fold.split(group_ids)):
        print(f"\nProcessing fold{fold_index}")
        train_groups = {group_ids[index] for index in train_indices}
        val_groups = {group_ids[index] for index in val_indices}

        for split_name, selected_groups in (("train", train_groups), ("val", val_groups)):
            selected_pairs = [
                pair for pair in segment_pairs if pair.group_id in selected_groups
            ]
            progress = tqdm(selected_pairs, desc=f"fold{fold_index} {split_name}")

            for pair in progress:
                ap_output_dir = (
                    config.output_root / "ap" / f"fold{fold_index}" / split_name / pair.target_label
                )
                lat_output_dir = (
                    config.output_root / "lat" / f"fold{fold_index}" / split_name / pair.target_label
                )
                ap_output_dir.mkdir(parents=True, exist_ok=True)
                lat_output_dir.mkdir(parents=True, exist_ok=True)

                total_clips += split_pair_into_clips(
                    pair=pair,
                    ap_output_dir=ap_output_dir,
                    lat_output_dir=lat_output_dir,
                    clip_seconds=config.clip_seconds,
                )

    print(f"\nFinished fold generation. total_clips={total_clips}")


if __name__ == "__main__":
    prepare_dataset(parse_args())
