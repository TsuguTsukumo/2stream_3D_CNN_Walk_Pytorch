from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rename segmented AP/LAT files to canonical names."
    )
    parser.add_argument("root_dir", help="Root directory that contains segment folders.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def rename_files(root_dir: str | Path) -> int:
    renamed_count = 0
    root_path = Path(root_dir)

    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.name.startswith("lat_"):
            new_name = "lat.mp4"
        elif file_path.name.startswith("ap_"):
            new_name = "ap.mp4"
        else:
            continue

        new_path = file_path.with_name(new_name)
        if file_path == new_path:
            continue

        file_path.rename(new_path)
        renamed_count += 1
        print(f"Renamed: {file_path} -> {new_path}")

    return renamed_count


def main(args: argparse.Namespace) -> None:
    renamed_count = rename_files(args.root_dir)
    print(f"Renamed {renamed_count} files.")


if __name__ == "__main__":
    main(parse_args())
