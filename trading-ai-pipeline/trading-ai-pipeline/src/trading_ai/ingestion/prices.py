"""Price ingestion: yfinance for real data, a seeded generator for offline testing."""
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def fetch_prices(tickers, start, end=None):
    """Daily OHLCV from Yahoo Finance (split/dividend adjusted)."""
    import yfinance as yf

    frames = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
        except Exception as ex:  # network, symbol issues
            log.warning("Failed to fetch %s: %s", t, ex)
            continue
        if h.empty:
            log.warning("No data for %s", t)
            continue
        h = h.reset_index()
        h.columns = [str(c).lower() for c in h.columns]
        date_col = "date" if "date" in h.columns else "datetime"
        h["date"] = pd.to_datetime(h[date_col]).dt.tz_localize(None).dt.normalize()
        h["ticker"] = t
        frames.append(h[COLUMNS])
    if not frames:
        raise RuntimeError("No price data fetched. Check network / tickers.")
    return pd.concat(frames, ignore_index=True)


def synthetic_prices(tickers, start, end, seed=42, phi=0.08):
    """Fake but realistic-ish prices (AR(1) returns, GBM-like). For plumbing tests only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    frames = []
    for t in tickers:
        eps = rng.normal(0, 0.012, n)
        r = np.zeros(n)
        for k in range(1, n):
            r[k] = 0.0003 + phi * r[k - 1] + eps[k]
        close = 100 * np.exp(np.cumsum(r))
        open_ = close * (1 + rng.normal(0, 0.002, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
        vol = rng.integers(1_000_000, 5_000_000, n) * (1 + np.abs(r) * 20)
        frames.append(
            pd.DataFrame(
                {"date": dates, "ticker": t, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
            )
        )
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def save_raw(df, folder: Path):
    """Raw layer is append-only: every run writes a new timestamped snapshot."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = folder / f"prices_{stamp}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_latest(folder: Path):
    files = sorted(Path(folder).glob("prices_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw price snapshots in {folder}. Run the ingest stage first.")
    df = pd.read_parquet(files[-1])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
