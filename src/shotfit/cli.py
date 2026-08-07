"""Command-line entrypoints for the reproducible pipeline."""

from __future__ import annotations

import argparse

from shotfit.evaluation import export_app_bundle
from shotfit.features import build_database
from shotfit.ingest import ingest_all
from shotfit.modeling import train_and_score


def main() -> None:
    parser = argparse.ArgumentParser(prog="shotfit")
    parser.add_argument("command", choices=("ingest", "build-features", "train", "export-app", "all"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.command in ("ingest", "all"):
        ingest_all(refresh=args.refresh)
    if args.command in ("build-features", "all"):
        build_database()
    if args.command in ("train", "all"):
        train_and_score()
    if args.command in ("export-app", "all"):
        export_app_bundle()


if __name__ == "__main__":
    main()

