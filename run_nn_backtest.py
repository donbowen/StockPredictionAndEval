"""Full expanding-window MLP backtest (specs chosen by run_nn_screen.py on the 2005 window), then evaluation
next to the corrected linear models from run_linear_backtest.py.

    python run_nn_backtest.py --data-dir <input_data> --run-id nn_v1 [--smoke] [--evaluate-only]
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

from stockpred.nn import MLPSpec, NNBacktest
from stockpred.nn_report import evaluate_nn
from stockpred.runner import Backtest, Config, MODEL_LABELS

S = MLPSpec
SPECS = [   # screened on 2005 validation (train < 2002, val 2002-2004); 5-seed ensembles
    S("MLP_rank_256x128x64_do0.3", (256, 128, 64), dropout=0.3, target="rank", n_seeds=5),
    S("MLP_rank_128x64", (128, 64), target="rank", n_seeds=5),
    S("MLP_huber_128x64", (128, 64), target="clip", loss="huber", n_seeds=5),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run-id", default="nn_v1")
    ap.add_argument("--linear-run", default="linear_v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    names = [s.name for s in SPECS]
    if a.smoke:
        cfg = Config(run_id=a.run_id + "_smoke", data_dir=a.data_dir, smoke_permnos=500, first_test_year=2018)
        NNBacktest(cfg, SPECS).run_nn(names)
        print("smoke OK")
        return
    cfg = Config(run_id=a.run_id, data_dir=a.data_dir)
    if not a.evaluate_only:
        NNBacktest(cfg, SPECS).run_nn(names)
    from stockpred import data as D
    root, lroot = Path(cfg.out_root) / cfg.run_id, Path(cfg.out_root) / a.linear_run
    perf, reg, ics, icy, tie, checks = evaluate_nn(root, names, lroot, ["OLS", "RidgeCV"], cfg.years(),
                                                   D.resolve_data_dir(cfg.data_dir))
    print(perf.round(3).to_string())
    print(ics.round(4).to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
