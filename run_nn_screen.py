"""Architecture / target screen for the MLP. Uses ONLY data before the screen year Y (train < Y-3,
validation Y-3..Y-1); the screen year's own returns are never loaded into any tensor.

    python run_nn_screen.py --data-dir <input_data> --year 2005
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from stockpred.nn import MLPSpec, NNBacktest
from stockpred.runner import Config

S = MLPSpec
STAGE1 = [   # width / depth at a neutral setting
    S("h32", (32,)), S("h64_32", (64, 32)), S("h128_64", (128, 64)), S("h256_128_64", (256, 128, 64)),
    S("h512_256_128_64", (512, 256, 128, 64)), S("h32_16_8", (32, 16, 8)),
    S("h128_64_bn", (128, 64), batchnorm=True),
]
TARGETS = [   # target / loss on the mid-size net
    S("t_raw", (128, 64), target="raw"), S("t_demean", (128, 64), target="cs_demean"),
    S("t_rank", (128, 64), target="rank"), S("t_clip_huber", (128, 64), loss="huber"),
]


def run(bt, d, specs, out, tag):
    rows = []
    for s in specs:
        t = time.time()
        _, diag = bt.train_spec(d, s, [1])
        r = {"stage": tag, "name": s.name, "hidden": list(s.hidden), "dropout": s.dropout, "wd": s.weight_decay,
             "lr": s.lr, "target": s.target, "loss": s.loss, "bn": s.batchnorm,
             "val_spearman_ic": diag["val_spearman_ic"], "val_pearson_ic": diag["val_pearson_ic_ensemble"],
             "best_epoch": diag["runs"][0]["best_epoch"], "epochs_run": diag["runs"][0]["epochs_run"],
             "params": diag["n_params"], "seconds": round(time.time() - t, 1)}
        rows.append(r)
        print(r, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--year", type=int, default=2005)
    ap.add_argument("--run-id", default="nn_screen")
    a = ap.parse_args()
    cfg = Config(run_id=a.run_id, data_dir=a.data_dir)
    bt = NNBacktest(cfg, [])
    d = bt.prepare(a.year, with_test=False)
    assert "Xte" not in d and "yte" not in d
    print(f"screen year {a.year}: train rows {len(d['Xtr']):,}, val rows {len(d['Xva']):,}, features {d['Xtr'].shape[1]}, "
          f"device {d['dev']}", flush=True)
    out = Path(cfg.out_root) / a.run_id / "screen.csv"
    r1 = run(bt, d, STAGE1, out, "arch")
    best_arch = r1.sort_values("val_spearman_ic", ascending=False).iloc[0]
    print("best arch:", best_arch["name"], flush=True)
    hid = tuple(best_arch["hidden"])
    bn = bool(best_arch["bn"])
    r2 = run(bt, d, TARGETS, str(out).replace("screen.csv", "screen_targets.csv"), "target")
    best_t = r2.sort_values("val_spearman_ic", ascending=False).iloc[0]
    print("best target:", best_t["name"], flush=True)
    tgt, loss = best_t["target"], best_t["loss"]
    reg = [S(f"{'x'.join(map(str, hid))}_do{do}_wd{wd}_lr{lr}", hid, dropout=do, weight_decay=wd, lr=lr,
             target=tgt, loss=loss, batchnorm=bn)
           for do, wd, lr in [(0.0, 1e-4, 1e-3), (0.1, 1e-4, 1e-3), (0.3, 1e-4, 1e-3), (0.1, 1e-2, 1e-3),
                              (0.1, 1e-4, 3e-4), (0.1, 1e-4, 3e-3)]]
    r3 = run(bt, d, reg, str(out).replace("screen.csv", "screen_reg.csv"), "reg")
    pd.concat([r1, r2, r3]).to_csv(Path(cfg.out_root) / a.run_id / "screen_all.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
