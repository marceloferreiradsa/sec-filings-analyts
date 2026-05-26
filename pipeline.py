"""
pipeline.py — Build orchestrator for SEC Filings Analyst.

Runs the full data pipeline or individual stages.
The actual implementation of each stage lives in stages/.

Usage:
  python pipeline.py                   run full pipeline
  python pipeline.py --only embed      run a single stage
  python pipeline.py --from chunk      run from this stage onwards
  python pipeline.py --status          check what has been built
"""

import argparse
import sys
from pathlib import Path

STAGES = ["ingest", "chunk", "embed", "index"]

_STATUS_PATHS = {
    "ingest": Path("data/raw"),
    "chunk":  Path("data/processed/chunks.json"),
    "embed":  Path("data/processed/embeddings.npy"),
    "index":  Path("data/index/index.faiss"),
}


def _run_stage(stage: str) -> None:
    print(f"\n[{stage.upper()}] Starting…")
    if stage == "ingest":
        from stages.ingest import main
    elif stage == "chunk":
        from stages.chunk import main
    elif stage == "embed":
        from stages.embed import main
    elif stage == "index":
        from stages.index import main
    main()
    print(f"[{stage.upper()}] Done.")


def _status() -> None:
    print("\nPipeline status")
    print("─" * 44)
    for stage, path in _STATUS_PATHS.items():
        exists = path.exists()
        mark = "✓" if exists else "✗"
        extra = ""
        if exists and path.is_file():
            mb = path.stat().st_size / 1e6
            extra = f"  ({mb:.1f} MB)"
        elif exists and path.is_dir():
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            extra = f"  ({n} files)"
        print(f"  {mark}  {stage:<8}  {path}{extra}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Build the SEC Filings Analyst index from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python pipeline.py                   full pipeline\n"
            "  python pipeline.py --only embed      single stage\n"
            "  python pipeline.py --from chunk      from stage onwards\n"
            "  python pipeline.py --status          show build status"
        ),
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--only",
        choices=STAGES,
        metavar="STAGE",
        help=f"Run only this stage. Choices: {', '.join(STAGES)}",
    )
    grp.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGES,
        metavar="STAGE",
        help="Run from this stage through the end of the pipeline.",
    )
    grp.add_argument(
        "--status",
        action="store_true",
        help="Show what has been built and exit.",
    )
    args = parser.parse_args()

    if args.status:
        _status()
        return

    if args.only:
        _run_stage(args.only)
        return

    to_run = STAGES[STAGES.index(args.from_stage):] if args.from_stage else STAGES
    print(f"Running stages: {' → '.join(to_run)}")
    for stage in to_run:
        _run_stage(stage)

    print(f"\n{'─' * 44}")
    print("Complete. Index ready at data/index/")


if __name__ == "__main__":
    main()
