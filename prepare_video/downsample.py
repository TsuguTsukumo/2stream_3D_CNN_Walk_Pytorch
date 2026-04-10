from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .pipeline_utils import reset_dir


@dataclass(frozen=True)
class DownsampleConfig:
    input_root: Path
    output_root: Path
    seed: int = 42


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Balance AP/LAT fold datasets without breaking synchronized pairs."
    )
    parser.add_argument("--input-root", default="/workspace/data/5folds")
    parser.add_argument("--output-root", default="/workspace/data/balanced_data")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: list[str] | None = None) -> DownsampleConfig:
    args = build_parser().parse_args(argv)
    return DownsampleConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        seed=args.seed,
    )


def build_relative_index(root_dir: Path) -> dict[str, Path]:
    return {
        path.relative_to(root_dir).as_posix(): path
        for path in sorted(root_dir.rglob("*.mp4"))
    }


def select_balanced_pairs(
    ap_index: dict[str, Path],
    lat_index: dict[str, Path],
    seed: int,
) -> dict[tuple[str, str, str], list[str]]:
    common_keys = sorted(set(ap_index) & set(lat_index))
    buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for relative_key in common_keys:
        relative_path = Path(relative_key)
        if len(relative_path.parts) < 4:
            continue

        fold_name, split_name, label_name = relative_path.parts[:3]
        buckets[(fold_name, split_name, label_name)].append(relative_key)

    split_to_labels: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    for (fold_name, split_name, label_name), keys in buckets.items():
        split_to_labels[(fold_name, split_name)][label_name] = sorted(keys)

    rng = random.Random(seed)
    selected: dict[tuple[str, str, str], list[str]] = {}

    for split_key, label_to_keys in split_to_labels.items():
        non_empty_counts = [len(keys) for keys in label_to_keys.values() if keys]
        if not non_empty_counts:
            continue

        target_count = min(non_empty_counts)
        for label_name, keys in label_to_keys.items():
            if len(keys) <= target_count:
                selected_keys = list(keys)
            else:
                selected_keys = sorted(rng.sample(keys, target_count))
            selected[(split_key[0], split_key[1], label_name)] = selected_keys

    return selected


def copy_selected_pairs(
    ap_index: dict[str, Path],
    lat_index: dict[str, Path],
    selected_pairs: dict[tuple[str, str, str], list[str]],
    output_root: Path,
) -> None:
    reset_dir(output_root)

    for relative_keys in selected_pairs.values():
        for relative_key in relative_keys:
            ap_source = ap_index[relative_key]
            lat_source = lat_index[relative_key]

            ap_destination = output_root / "ap" / relative_key
            lat_destination = output_root / "lat" / relative_key

            ap_destination.parent.mkdir(parents=True, exist_ok=True)
            lat_destination.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(ap_source, ap_destination)
            shutil.copy2(lat_source, lat_destination)


def downsample_dataset(config: DownsampleConfig) -> None:
    ap_root = config.input_root / "ap"
    lat_root = config.input_root / "lat"
    if not ap_root.is_dir() or not lat_root.is_dir():
        raise FileNotFoundError(
            f"Expected AP/LAT roots at {ap_root} and {lat_root}."
        )

    ap_index = build_relative_index(ap_root)
    lat_index = build_relative_index(lat_root)
    selected_pairs = select_balanced_pairs(
        ap_index=ap_index,
        lat_index=lat_index,
        seed=config.seed,
    )
    copy_selected_pairs(
        ap_index=ap_index,
        lat_index=lat_index,
        selected_pairs=selected_pairs,
        output_root=config.output_root,
    )

    print("Balanced dataset created successfully.")
    for bucket, selected in sorted(selected_pairs.items()):
        print(f"{bucket}: {len(selected)}")


if __name__ == "__main__":
    downsample_dataset(parse_args())
