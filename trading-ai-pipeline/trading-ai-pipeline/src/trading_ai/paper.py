"""Paper trading: log each day's prediction with a timestamp, settle it once the next close is known."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .features.build import feature_columns
from .models.train import make_model

LOG_COLUMNS = ["pred_date", "ticker", "p_up", "model", "logged_at", "status", "actual_ret", "correct"]


def _read_log(path):
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, parse_dates=["pred_date"])


def log_predictions(dataset, cfg, log_path):
    """Fit on every labelled row, predict the latest (unlabelled) rows, append new ones to the log."""
    feats = feature_columns(dataset)
    labelled = dataset.dropna(subset=["target"])
    latest = dataset[dataset["date"] == dataset["date"].max()]
    name = cfg["paper"]["model"]
    model = make_model(name, cfg["model"]["seed"]).fit(labelled[feats], labelled["target"])

    new = pd.DataFrame(
        {
            "pred_date": latest["date"].values,
            "ticker": latest["ticker"].values,
            "p_up": model.predict_proba(latest[feats])[:, 1],
            "model": name,
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "open",
            "actual_ret": np.nan,
            "correct": np.nan,
        }
    )
    log = _read_log(log_path)
    key = ["pred_date", "ticker", "model"]
    if not log.empty:
        seen = log[key].astype(str).agg("|".join, axis=1)
        new = new[~new[key].astype(str).agg("|".join, axis=1).isin(seen)]
    log = pd.concat([log, new], ignore_index=True) if not new.empty else log
    return log


def settle(log, prices):
    """Fill actual_ret / correct for open predictions whose next trading day has now happened."""
    px = prices.pivot(index="date", columns="ticker", values="close")
    for i, row in log[log["status"] == "open"].iterrows():
        s = px[row["ticker"]].dropna()
        pos = s.index.searchsorted(row["pred_date"])
        if pos + 1 < len(s) and s.index[pos] == row["pred_date"]:
            ret = s.iloc[pos + 1] / s.iloc[pos] - 1
            log.loc[i, "actual_ret"] = ret
            log.loc[i, "correct"] = float((row["p_up"] > 0.5) == (ret > 0))
            log.loc[i, "status"] = "settled"
    return log


def run(dataset, prices, cfg):
    path = cfg["paths"]["paper"] / "predictions_log.csv"
    log = log_predictions(dataset, cfg, path)
    log = settle(log, prices)
    log.to_csv(path, index=False)
    return log
