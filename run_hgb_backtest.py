"""Expanding-window gradient boosting backtest: next-month return, within-month rank, and a quintile classifier
(plus OLS on the rank target, to separate the effect of the target from the effect of the trees).

    python run_hgb_backtest.py --data-dir <input_data> --cache-dir <work_cache> --run-id hgb_v1 [--smoke] [--evaluate-only]
"""
import argparse
import time
from pathlib import Path

from stockpred import data as D
from stockpred.hgb import HGBBacktest, HGBSpec
from stockpred.nn_report import evaluate_nn
from stockpred.runner import Config

SPECS = [
    HGBSpec("HGB-return", target="return"),
    HGBSpec("HGB-rank", target="rank"),
    HGBSpec("HGB-quintile", target="quintile"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cache-dir", default="work_cache")
    ap.add_argument("--run-id", default="hgb_v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--linear-run", default="linear_v1")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    kw = dict(smoke_permnos=500, first_test_year=2018) if a.smoke else {}
    cfg = Config(run_id=a.run_id + ("_smoke" if a.smoke else ""), data_dir=a.data_dir, cache_dir=a.cache_dir, **kw)
    if not a.evaluate_only:
        with threadpool_limits(limits=a.threads):
            HGBBacktest(cfg, SPECS).run_hgb()
    if not a.smoke:
        names = [s.name for s in SPECS] + ["OLS-rank"]
        root, lroot = Path(cfg.out_root) / cfg.run_id, Path(cfg.out_root) / a.linear_run
        perf, _, ics, _, _, _ = evaluate_nn(root, names, lroot, ["OLS", "RidgeCV"], cfg.years(),
                                            D.resolve_data_dir(cfg.data_dir))
        print(perf.round(3).to_string())
        print(ics.round(4).to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
