from __future__ import annotations

import sys
from textwrap import dedent


def build_help_text() -> str:
    return dedent(
        """
        Usage:
          python -m prepare_video <command> [options]

        Commands:
          preprocess   Detect, crop, and segment synchronized raw videos
          split        Split synchronized AP/LAT segments into fold datasets
          balance      Downsample fold datasets while preserving AP/LAT pairs
          rename       Rename per-segment AP/LAT files to canonical names
        """
    ).strip()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help"}:
        print(build_help_text())
        return

    command, *command_args = argv

    if command == "preprocess":
        from .preprocess import parse_args as parse_preprocess_args, preprocess_dataset

        preprocess_dataset(parse_preprocess_args(command_args))
        return
    if command == "split":
        from .split_cross import parse_args as parse_split_args, prepare_dataset

        prepare_dataset(parse_split_args(command_args))
        return
    if command == "balance":
        from .downsample import downsample_dataset, parse_args as parse_balance_args

        downsample_dataset(parse_balance_args(command_args))
        return
    if command == "rename":
        from .rename import main as rename_main, parse_args as parse_rename_args

        rename_main(parse_rename_args(command_args))
        return

    raise SystemExit(f"Unknown command: {command}\n\n{build_help_text()}")
