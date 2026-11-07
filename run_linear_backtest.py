"""Reproduce the corrected linear-model backtest.

    python run_linear_backtest.py --data-dir <input_data> --run-id linear_v1

Stages (all resumable; a fit is skipped only if its saved config/data/code hashes match):
  1. smoke   - tiny end-to-end check (separate output dir) incl. all five models + evaluation
  2. OLS     - full annual backtest
  3. Ridge, RidgeCV, Lasso, LassoCV - full annual backtest (share one preprocessing pass per year)
  4. evaluate - portfolios, tie diagnostics, performance tables, comparison with the original results

Neural-network / boosting models are deliberately NOT run here; add them to
``stockpred.runner.make_estimator`` / ``MODEL_ORDER`` later.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from stockpred.runner import Backtest, Config
from stockpred.report import evaluate_run

ALL = ["OLS", "Ridge", "RidgeCV", "Lasso", "LassoCV"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run-id", default="linear_v1")
    ap.add_argument("--out-root", default="output_corrected")
    ap.add_argument("--cache-dir", default="work_cache")
    ap.add_argument("--blas-threads", type=int, default=12)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--only", nargs="*", choices=ALL, help="run only these models (default: staged OLS then the rest)")
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()

    base = Config(run_id=a.run_id, out_root=a.out_root, cache_dir=a.cache_dir, data_dir=a.data_dir,
                  blas_threads=a.blas_threads)
    t0 = time.time()
    if not a.skip_smoke and not a.evaluate_only:
        smoke = replace(base, run_id=a.run_id + "_smoke", smoke_permnos=500, first_test_year=2018)
        print("== smoke test (500 permnos, test years 2018-2019, all five models) ==", flush=True)
        bt = Backtest(smoke)
        bt.run(ALL)
        evaluate_run(smoke, starts=("2018-01",), smoke=True)
    bt = Backtest(base)
    if not a.evaluate_only:
        if a.only:
            bt.run(a.only, a.years)
        else:
            for stage in (["OLS"], ["Ridge", "RidgeCV", "Lasso", "LassoCV"]):
                print(f"== stage {stage} ==", flush=True)
                bt.run(stage, a.years)
    print("== evaluate ==", flush=True)
    evaluate_run(base)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
