"""Feature engineering. Every feature at row (date=t) uses ONLY data up to the close of t.

Target: does the close of t+1 beat the close of t?  (next_ret > 0)
The most recent row per ticker has no target yet; it is kept for live prediction.
"""
import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "rsi_14",
    "macd_hist", "dist_ma20", "vol_chg", "range_pct",
]
NEWS_FEATURES = [
    "mkt_sent", "mkt_count", "tkr_sent", "tkr_count", "mkt_sent_3d", "tkr_sent_3d",
]


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)


def price_features(g):
    g = g.sort_values("date").copy()
    c = g["close"]
    r = c.pct_change()
    g["ret_1"] = r
    g["ret_5"] = c.pct_change(5)
    g["ret_10"] = c.pct_change(10)
    g["vol_10"] = r.rolling(10).std()
    g["vol_20"] = r.rolling(20).std()
    g["rsi_14"] = rsi(c, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    g["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["vol_chg"] = np.log(g["volume"] / g["volume"].rolling(20).mean())
    g["range_pct"] = (g["high"] - g["low"]) / c
    g["next_ret"] = c.shift(-1) / c - 1
    g["target"] = (g["next_ret"] > 0).astype(float).where(g["next_ret"].notna())
    return g.replace([np.inf, -np.inf], np.nan)


def feature_columns(df):
    return [c for c in PRICE_FEATURES + NEWS_FEATURES if c in df.columns]


def build_dataset(prices, news_agg=None):
    df = pd.concat([price_features(g) for _, g in prices.groupby("ticker")], ignore_index=True)

    if news_agg is not None:
        mkt, tkr = news_agg
        df = df.merge(mkt, on="date", how="left").merge(tkr, on=["date", "ticker"], how="left")
        for c in ("mkt_count", "tkr_count"):
            df[c] = df[c].fillna(0.0)
        for c in ("mkt_sent", "tkr_sent"):
            df[c] = df[c].fillna(0.0)
        df = df.sort_values(["ticker", "date"])
        for c in ("mkt_sent", "tkr_sent"):
            df[c + "_3d"] = df.groupby("ticker")[c].transform(lambda s: s.rolling(3, min_periods=1).mean())

    feats = feature_columns(df)
    df = df.dropna(subset=feats).sort_values(["date", "ticker"]).reset_index(drop=True)
    return df
