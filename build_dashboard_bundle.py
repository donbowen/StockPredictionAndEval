"""Build the files the Streamlit dashboard reads for the corrected results.

    python build_dashboard_bundle.py --data-dir input_data --cache-dir work_cache

Needs the predictions written by run_linear_backtest.py (linear_v1), run_nn_backtest.py (nn_v1) and
run_hgb_backtest.py (hgb_v1) under output_corrected/.  Writes output_corrected/dashboard/.  The trading cost
tables also need the input data (bid-ask spreads and dollar volume); --no-costs skips them.
"""
import argparse
from pathlib import Path

from stockpred import data as D
from stockpred.dashboard_bundle import build_bundle, cost_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="output_corrected")
    ap.add_argument("--first-year", type=int, default=2000)
    ap.add_argument("--last-year", type=int, default=2019)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cache-dir", default="work_cache")
    ap.add_argument("--no-costs", action="store_true")
    a = ap.parse_args()
    years = list(range(a.first_year, a.last_year + 1))
    costs = None
    if not a.no_costs:
        panel = D.get_panel(D.resolve_data_dir(a.data_dir), Path(a.cache_dir))[0]
        costs = cost_inputs(panel[panel["date"].dt.year.isin(years)])
    idx = build_bundle(Path(a.out_root), Path(a.out_root) / "dashboard", years, costs)
    print("wrote", Path(a.out_root) / "dashboard", "models:", ", ".join(idx))


if __name__ == "__main__":
    main()
