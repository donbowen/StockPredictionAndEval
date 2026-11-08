"""Build the files the Streamlit dashboard reads for the corrected results.

    python build_dashboard_bundle.py

Needs the predictions written by run_linear_backtest.py (linear_v1) and run_nn_backtest.py (nn_v1)
under output_corrected/.  Writes output_corrected/dashboard/.
"""
import argparse
from pathlib import Path

from stockpred.dashboard_bundle import build_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="output_corrected")
    ap.add_argument("--first-year", type=int, default=2000)
    ap.add_argument("--last-year", type=int, default=2019)
    a = ap.parse_args()
    idx = build_bundle(Path(a.out_root), Path(a.out_root) / "dashboard", list(range(a.first_year, a.last_year + 1)))
    print("wrote", Path(a.out_root) / "dashboard", "models:", ", ".join(idx))


if __name__ == "__main__":
    main()
