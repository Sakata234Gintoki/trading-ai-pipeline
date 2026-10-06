"""Pipeline stages. Each stage is a plain function so Airflow, the CLI and tests can all call it."""
import json
import logging

import pandas as pd

from . import paper
from .backtest import engine as E
from .config import load_config
from .features import build as B
from .features import news_features as NF
from .ingestion import news as N
from .ingestion import prices as P
from .models import train as T

log = logging.getLogger(__name__)


def _cfg(cfg):
    return cfg or load_config()


def ingest_prices(cfg=None):
    cfg = _cfg(cfg)
    if cfg["synthetic_mode"]:
        s = cfg["synthetic"]
        df = P.synthetic_prices(cfg["tickers"], cfg["start_date"], s["end_date"], s["seed"], s["ar_phi"])
    else:
        df = P.fetch_prices(cfg["tickers"], cfg["start_date"])
    path = P.save_raw(df, cfg["paths"]["raw"] / "prices")
    log.info("Saved %d price rows -> %s", len(df), path)


def ingest_news(cfg=None):
    cfg = _cfg(cfg)
    if cfg["synthetic_mode"]:
        s = cfg["synthetic"]
        df = N.synthetic_news(cfg["start_date"], s["end_date"], cfg["news"]["aliases"], s["seed"])
    else:
        df = N.fetch_rss(cfg["news"]["feeds"])
    path = N.save_raw(df, cfg["paths"]["raw"] / "news")
    log.info("Saved %d news rows -> %s", len(df), path)


def build_features(cfg=None):
    cfg = _cfg(cfg)
    prices = P.load_latest(cfg["paths"]["raw"] / "prices")
    agg = None
    if cfg["features"]["use_news"]:
        news = N.load_all(cfg["paths"]["raw"] / "news")
        if news.empty:
            log.warning("use_news is on but no news collected yet; building price-only features.")
        else:
            news["sentiment"] = NF.get_scorer(cfg["features"]["sentiment"])(news["title"].tolist())
            ncfg = cfg["news"]
            agg = NF.aggregate(
                news, prices["date"].unique(), ncfg["aliases"], ncfg["market_close_ist"], ncfg["timezone"]
            )
    df = B.build_dataset(prices, agg)
    out = cfg["paths"]["processed"] / "dataset.parquet"
    df.to_parquet(out, index=False)
    log.info("Dataset: %d rows, %d features -> %s", len(df), len(B.feature_columns(df)), out)


def train_and_backtest(cfg=None):
    cfg = _cfg(cfg)
    df = pd.read_parquet(cfg["paths"]["processed"] / "dataset.parquet")
    feats = B.feature_columns(df)
    bt = cfg["backtest"]
    metrics, curves, first_preds = [], {}, None

    for name in cfg["model"]["models"]:
        preds = T.walk_forward(df, feats, name, cfg)
        preds.to_parquet(cfg["paths"]["processed"] / f"predictions_{name}.parquet", index=False)
        m, daily = E.evaluate(name, preds, bt["threshold"], bt["cost_bps"])
        metrics.append(m)
        curves[name] = daily
        first_preds = preds if first_preds is None else first_preds

    for kind in ("buy_hold", "prev_day"):
        m, daily = E.evaluate(kind, E.baseline_preds(first_preds, kind), bt["threshold"], bt["cost_bps"])
        metrics.append(m)
        curves[kind] = daily

    equity = (1 + pd.DataFrame(curves)).cumprod()
    equity.index.name = "date"
    equity.reset_index().to_csv(cfg["paths"]["processed"] / "equity_curves.csv", index=False)
    with open(cfg["paths"]["processed"] / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    table = pd.DataFrame(metrics).set_index("model")
    cols = ["n_predictions", "accuracy", "acc_ci95_low", "acc_ci95_high", "auc", "total_return", "sharpe", "max_drawdown"]
    log.info("\n%s", table[cols].round(4).to_string())
    return table


def paper_trade(cfg=None):
    cfg = _cfg(cfg)
    df = pd.read_parquet(cfg["paths"]["processed"] / "dataset.parquet")
    prices = P.load_latest(cfg["paths"]["raw"] / "prices")
    result = paper.run(df, prices, cfg)
    log.info("Paper log: %d rows (%d settled)", len(result), int((result["status"] == "settled").sum()))
    return result


def run_all(cfg=None):
    cfg = _cfg(cfg)
    ingest_prices(cfg)
    ingest_news(cfg)
    build_features(cfg)
    train_and_backtest(cfg)
    paper_trade(cfg)


STAGES = {
    "ingest": lambda cfg: (ingest_prices(cfg), ingest_news(cfg)),
    "features": build_features,
    "train": train_and_backtest,
    "paper": paper_trade,
    "all": run_all,
}
