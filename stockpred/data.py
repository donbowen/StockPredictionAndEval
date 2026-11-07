"""Data loading and panel construction.

Timing convention (the single source of truth for the whole project)
--------------------------------------------------------------------
Every row is a stock-month keyed by ``(permno, date)`` where ``date`` is the
**return month** (anchored to the 28th so it exists in every month).

* ``signal_date``     - month in which the OAP signals were *measured*.
* ``ff_source_date``  - month in which the predictor factors were *measured*.
* ``date``            - month whose stock return is being predicted.

Both predictor families satisfy ``date == signal_date + 1 month``; i.e. what is
measured in month t predicts the return in month t+1 (Dec -> Jan crosses the
year boundary).  The factors used for *performance evaluation* (``ff_eval``) are
contemporaneous and are never shifted.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_COLS = ["mkt_excess", "SMB", "HML", "RMW", "CMA", "Mom"]
EVAL_FACTOR_COLS = FACTOR_COLS + ["RF"]
ANCHOR_DAY = 28


# ---------------------------------------------------------------- timing ---
def anchor_month(year, month) -> pd.DatetimeIndex:
    """Datetimes on the 28th of the given year/month (vectorised; positional, no index alignment)."""
    return pd.DatetimeIndex(pd.to_datetime(
        {"year": np.asarray(year), "month": np.asarray(month), "day": ANCHOR_DAY}
    ))


def yyyymm_to_anchor(yyyymm) -> pd.DatetimeIndex:
    v = np.asarray(yyyymm).astype(np.int64)
    return anchor_month(v // 100, v % 100)


def next_month(dates):
    """Return month that a predictor measured on ``dates`` is used to predict."""
    return dates + pd.DateOffset(months=1)


def month_index(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (d.year * 12 + d.month).to_numpy()


def lag_factors_for_prediction(ff_eval: pd.DataFrame) -> pd.DataFrame:
    """Predictor version of the factors: realised in month t, used for t+1.

    ``ff_eval`` (contemporaneous, used for performance regressions) is NOT
    modified.  The returned frame has ``date`` = measurement month + 1 month.
    (The old notebook subtracted a month here, which put month t+1 factor
    realisations into month t's predictor row.)
    """
    out = ff_eval[["date"] + FACTOR_COLS].copy()
    out["ff_source_date"] = out["date"]
    out["date"] = next_month(out["date"])
    return out


def assert_timing(panel: pd.DataFrame) -> dict:
    """Hard timing assertions + worked examples (incl. Dec -> Jan)."""
    d = month_index(panel["date"]) - month_index(panel["signal_date"])
    assert (d == 1).all(), "signal_date must be exactly one month before date"
    f = month_index(panel["date"]) - month_index(panel["ff_source_date"])
    assert (f == 1).all(), "ff_source_date must be exactly one month before date"
    assert (panel["signal_date"] == panel["ff_source_date"]).all()
    ex = {}
    dec = panel[panel["signal_date"].dt.month == 12].head(1)
    if len(dec):
        ex["december_to_january"] = {
            "signal_date": str(dec["signal_date"].iloc[0].date()),
            "ff_source_date": str(dec["ff_source_date"].iloc[0].date()),
            "return_date": str(dec["date"].iloc[0].date()),
        }
        assert dec["date"].iloc[0].month == 1
        assert dec["date"].iloc[0].year == dec["signal_date"].iloc[0].year + 1
    first = panel.head(1)
    ex["first_row"] = {
        "signal_date": str(first["signal_date"].iloc[0].date()),
        "return_date": str(first["date"].iloc[0].date()),
    }
    return ex


def assert_factor_values_lagged(panel: pd.DataFrame, ff_eval: pd.DataFrame) -> dict:
    """Value-level check: the predictor factor on a row equals ff_eval at the *prior* month."""
    ev = ff_eval.set_index("date")[FACTOR_COLS]
    chk = panel.drop_duplicates("date")[["date", "ff_source_date"] + FACTOR_COLS].set_index("date")
    src = ev.loc[chk["ff_source_date"].to_numpy()]
    assert np.allclose(chk[FACTOR_COLS].to_numpy(), src.to_numpy(), equal_nan=False)
    d = panel["date"].iloc[0]
    return {"example_return_month": str(d.date()), "factor_measured_in": str(
        panel.loc[panel["date"] == d, "ff_source_date"].iloc[0].date())}


# ------------------------------------------------------------ identifiers ---
def _sha256(path: Path, block=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def data_identifiers(data_dir: Path) -> dict:
    """Content identifiers for the raw inputs (zip: stored CRC32, no re-read)."""
    z = data_dir / "signed_predictors_dl_wide.zip"
    with zipfile.ZipFile(z) as zf:
        info = zf.infolist()[0]
        zid = {"member": info.filename, "size": info.file_size, "crc32": f"{info.CRC:08x}"}
    return {
        "crsp_data.csv": _sha256(data_dir / "crsp_data.csv"),
        "ff_factors.csv": _sha256(data_dir / "ff_factors.csv"),
        "signal_documentation.csv": _sha256(data_dir / "signal_documentation.csv"),
        "signed_predictors_dl_wide.zip": zid,
    }


REQUIRED_INPUTS = ["crsp_data.csv", "ff_factors.csv", "signal_documentation.csv",
                   "signed_predictors_dl_wide.zip"]


def resolve_data_dir(arg: str | None) -> Path:
    """--data-dir, $STOCKPRED_DATA_DIR, ./input_data, then the primary checkout's input_data."""
    cands = []
    if arg:
        cands.append(Path(arg))
    if os.environ.get("STOCKPRED_DATA_DIR"):
        cands.append(Path(os.environ["STOCKPRED_DATA_DIR"]))
    cands.append(Path("input_data"))
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        cands.append(Path(common).resolve().parent / "input_data")
    except Exception:
        pass
    for c in cands:
        if all((c / n).exists() for n in REQUIRED_INPUTS):
            return c.resolve()
    missing = {str(c): [n for n in REQUIRED_INPUTS if not (c / n).exists()] for c in cands}
    raise FileNotFoundError("Required input files missing. Searched: " + json.dumps(missing, indent=1))


# ---------------------------------------------------------------- loaders ---
def load_crsp(path: Path, min_year: int, max_year: int) -> pd.DataFrame:
    crsp = pd.read_csv(path, usecols=["permno", "date", "ret"])
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp = crsp[crsp["ret"].notna() & crsp["date"].dt.year.between(min_year, max_year)].copy()
    crsp["date"] = anchor_month(crsp["date"].dt.year, crsp["date"].dt.month)
    assert not crsp.duplicated(["permno", "date"]).any()
    return crsp.reset_index(drop=True)  # ret in PERCENT


def load_ff_eval(path: Path) -> pd.DataFrame:
    """Contemporaneous factors (percent) for performance regressions."""
    ff = pd.read_csv(path, parse_dates=["date"])
    ff["date"] = anchor_month(ff["date"].dt.year, ff["date"].dt.month)
    ff = ff[["date"] + EVAL_FACTOR_COLS].sort_values("date").reset_index(drop=True)
    assert ff["date"].is_unique
    return ff


def feature_lists(doc_path: Path, available: list[str]) -> tuple[list[str], list[str]]:
    doc = pd.read_csv(doc_path)[["Acronym", "Cat.Form"]].drop_duplicates()
    avail = set(available)
    cont = [c for c in doc.loc[doc["Cat.Form"] == "continuous", "Acronym"] if c in avail]
    disc = [c for c in doc.loc[doc["Cat.Form"] == "discrete", "Acronym"] if c in avail]
    return cont, disc


def load_signals(zip_path: Path, keep_permnos, first_signal_ym: int, last_signal_ym: int,
                 min_nonmissing: float = 0.10, chunksize: int = 250_000) -> pd.DataFrame:
    """Read the (6GB) wide OAP file in chunks as float32.

    Row filter (row-local, so no cross-sample information is used): keep rows
    with >= ``min_nonmissing`` of the signal columns observed.
    """
    cols = pd.read_csv(zip_path, nrows=0).columns.tolist()
    sig_cols = [c for c in cols if c not in ("permno", "yyyymm")]
    dtypes = {c: np.float32 for c in sig_cols}
    dtypes.update(permno=np.int64, yyyymm=np.int64)
    keep_permnos = set(keep_permnos)
    parts = []
    for ch in pd.read_csv(zip_path, dtype=dtypes, chunksize=chunksize):
        ch = ch[ch["yyyymm"].between(first_signal_ym, last_signal_ym) & ch["permno"].isin(keep_permnos)]
        if len(ch) == 0:
            continue
        ch = ch[ch[sig_cols].notna().mean(axis=1) >= min_nonmissing]
        parts.append(ch)
    sig = pd.concat(parts, ignore_index=True)
    sig["signal_date"] = yyyymm_to_anchor(sig["yyyymm"])
    sig["date"] = next_month(sig["signal_date"])  # signals measured in t predict returns in t+1
    return sig.drop(columns="yyyymm")


def build_panel(data_dir: Path, min_return_year: int, max_return_year: int,
                min_nonmissing: float = 0.10):
    """Return (panel, continuous_cols, discrete_cols, ff_eval).

    Only return months <= ``max_return_year`` are ever loaded, so the 2021+
    holdout (and the 2020 buffer year) cannot influence any step.
    """
    crsp = load_crsp(data_dir / "crsp_data.csv", min_return_year, max_return_year)
    ff_eval = load_ff_eval(data_dir / "ff_factors.csv")
    fm = crsp["date"].min() - pd.DateOffset(months=1)  # measurement month of earliest return month
    lm = crsp["date"].max() - pd.DateOffset(months=1)
    sig = load_signals(data_dir / "signed_predictors_dl_wide.zip", crsp["permno"].unique(),
                       fm.year * 100 + fm.month, lm.year * 100 + lm.month, min_nonmissing)
    cont, disc = feature_lists(data_dir / "signal_documentation.csv", sig.columns.tolist())
    sig = sig[["permno", "date", "signal_date"] + cont + disc]

    panel = crsp.merge(sig, on=["permno", "date"], how="inner", validate="one_to_one")
    panel = panel.merge(lag_factors_for_prediction(ff_eval), on="date", how="left", validate="many_to_one")
    assert panel[FACTOR_COLS].notna().all().all(), "predictor factors missing for some return months"
    panel = panel.sort_values(["date", "permno"]).reset_index(drop=True)
    assert not panel.duplicated(["permno", "date"]).any()
    assert panel["ret"].notna().all()
    assert panel["date"].dt.year.max() <= max_return_year
    return panel, cont + FACTOR_COLS, disc, ff_eval


def get_panel(data_dir: Path, cache_dir: Path, min_return_year=1975, max_return_year=2019,
              min_nonmissing=0.10, use_cache=True):
    """Build (or load from a content-keyed parquet cache) the modelling panel."""
    ids = data_identifiers(data_dir)
    key = hashlib.sha256(json.dumps([ids, min_return_year, max_return_year, min_nonmissing],
                                    sort_keys=True).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    f, meta = cache_dir / f"panel_{key}.parquet", cache_dir / f"panel_{key}.json"
    if use_cache and f.exists() and meta.exists():
        m = json.loads(meta.read_text())
        return pd.read_parquet(f), m["continuous"], m["discrete"], load_ff_eval(data_dir / "ff_factors.csv"), key, ids
    panel, cont, disc, ff_eval = build_panel(data_dir, min_return_year, max_return_year, min_nonmissing)
    panel.to_parquet(f)
    meta.write_text(json.dumps({"continuous": cont, "discrete": disc, "identifiers": ids}))
    return panel, cont, disc, ff_eval, key, ids
