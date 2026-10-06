"""Backtest: long-or-cash per ticker, equal weight, transaction costs included."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def baseline_preds(preds, kind):
    """Dumb strategies every model must beat out-of-sample."""
    b = preds.copy()
    if kind == "buy_hold":
        b["p_up"] = 1.0
    elif kind == "prev_day":
        b["p_up"] = (b["ret_1"] > 0).astype(float)
    else:
        raise ValueError(kind)
    return b


def backtest(preds, threshold=0.5, cost_bps=10.0):
    """Position set at close of t (P(up) > threshold -> long) earns next_ret (t -> t+1).

    Each unit of position change pays cost_bps. Idle capital earns 0.
    Returns the daily portfolio return series.
    """
    p = preds.pivot(index="date", columns="ticker", values="p_up")
    r = preds.pivot(index="date", columns="ticker", values="next_ret").fillna(0.0)
    pos = (p > threshold).astype(float)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0].abs()
    net = pos * r - turnover * cost_bps / 1e4
    return net.mean(axis=1)


def classification_metrics(preds):
    y = preds["target"]
    yhat = (preds["p_up"] > 0.5).astype(float)
    n = len(y)
    acc = float((y == yhat).mean())
    se = float(np.sqrt(acc * (1 - acc) / n))
    auc = np.nan
    if y.nunique() > 1 and preds["p_up"].nunique() > 1:
        auc = float(roc_auc_score(y, preds["p_up"]))
    return {
        "n_predictions": int(n),
        "accuracy": acc,
        "acc_ci95_low": acc - 1.96 * se,
        "acc_ci95_high": acc + 1.96 * se,
        "base_rate_up": float(y.mean()),
        "auc": auc,
    }


def return_metrics(daily):
    equity = (1 + daily).cumprod()
    ann_vol = daily.std() * np.sqrt(252)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "ann_return": float(daily.mean() * 252),
        "sharpe": float(daily.mean() * 252 / ann_vol) if ann_vol > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def evaluate(name, preds, threshold, cost_bps):
    daily = backtest(preds, threshold, cost_bps)
    metrics = {"model": name, **classification_metrics(preds), **return_metrics(daily)}
    return metrics, daily
