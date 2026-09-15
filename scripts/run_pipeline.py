#!/usr/bin/env python
"""
CLI entrypoint for the full CRM analytics pipeline.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --no-market-basket
    python scripts/run_pipeline.py --excel data/raw/online_retail_II.xlsx

Regenerates everything under data/processed/ and models/ from the raw
Excel workbook. Safe to re-run: all outputs are overwritten deterministically
(random_state is fixed everywhere).
"""
import argparse
import sys
from pathlib import Path

# Make `src` importable when this script is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.pipeline import run_full_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Online Retail II CRM pipeline.")
    parser.add_argument(
        "--excel",
        default=str(config.RAW_EXCEL_PATH),
        help="Path to the Online Retail II .xlsx workbook.",
    )
    parser.add_argument(
        "--no-market-basket",
        action="store_true",
        help="Skip the (slow, optional) market-basket association-rule step.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the step-by-step cleaning summary.",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"ERROR: raw data not found at {excel_path}", file=sys.stderr)
        print("Download Online Retail II from:", file=sys.stderr)
        print("  https://archive.ics.uci.edu/dataset/502/online+retail+ii", file=sys.stderr)
        print(f"and place the .xlsx at {config.RAW_EXCEL_PATH}", file=sys.stderr)
        return 1

    run_full_pipeline(
        excel_path=excel_path,
        run_market_basket=not args.no_market_basket,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
