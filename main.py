#!/usr/bin/env python3
"""
CLI entrypoint for the AI Resume Screening & Ranking System.

Usage:
    python main.py --input ./resumes --output ./output/results.json
    python main.py --no-llm --input ./resumes
    python main.py --verbose --csv
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Insert src/ on sys.path so screener is importable
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def main() -> int:
    """Parse arguments and run the screening pipeline."""
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory containing resumes (default: from config)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for results.json (default: from config)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without LLM (deterministic fallback analyzer only)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent operations",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write results.csv alongside results.json",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load settings
    from screener.config import load_settings
    settings = load_settings()

    # Override settings from CLI args
    if args.max_concurrency:
        settings.max_concurrency = args.max_concurrency

    input_dir = args.input or settings.default_input_dir
    output_path = args.output or settings.default_output_path
    use_llm = not args.no_llm

    # Validate input directory
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error("Input directory does not exist: %s", input_path)
        return 1
    if not input_path.is_dir():
        logging.error("Input path is not a directory: %s", input_path)
        return 1

    # Run the pipeline
    from screener.pipeline import run_screening
    from screener.report import print_terminal_summary, write_csv, write_json

    logging.info(
        "Starting screening: input=%s, output=%s, use_llm=%s",
        input_path, output_path, use_llm,
    )

    run = asyncio.run(run_screening(input_dir=input_path, settings=settings, use_llm=use_llm))

    # Write output
    write_json(run, output_path)

    if args.csv:
        write_csv(run, output_path)

    # Print terminal summary
    print_terminal_summary(run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
