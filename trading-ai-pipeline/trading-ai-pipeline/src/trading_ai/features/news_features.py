"""News -> sentiment -> per-day features, with a strict "no peeking" timing rule.

Rule: a headline published at/after market close (default 15:30 IST) is only
usable for the NEXT trading day's decision. Weekends/holidays roll forward to
the next date that exists in the price calendar.
"""
import re

import numpy as np
import pandas as pd

_POS = {
    "surge", "surges", "gain", "gains", "beat", "beats", "growth", "record", "profit", "upgrade",
    "rally", "rallies", "strong", "rise", "rises", "jump", "jumps", "bullish", "outperform", "robust",
}
_NEG = {
    "fall", "falls", "drop", "drops", "slump", "slumps", "weak", "loss", "losses", "downgrade",
    "probe", "fraud", "concern", "concerns", "warns", "decline", "declines", "bearish", "miss", "misses", "lower",
}


def lexicon_scorer(texts):
    """Dumb but dependency-free sentiment in [-1, 1]. Good enough to test the plumbing."""
    out = []
    for t in texts:
        words = re.findall(r"[a-z]+", str(t).lower())
        pos = sum(w in _POS for w in words)
        neg = sum(w in _NEG for w in words)
        out.append((pos - neg) / (pos + neg) if pos + neg else 0.0)
    return np.array(out, dtype=float)


def finbert_scorer(texts, batch_size=32):
    """P(positive) - P(negative) from ProsusAI/finbert. Needs: pip install transformers torch."""
    from transformers import pipeline

    pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    scores = []
    for i in range(0, len(texts), batch_size):
        for result in pipe(list(texts[i : i + batch_size])):
            d = {r["label"].lower(): r["score"] for r in result}
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
    return np.array(scores, dtype=float)


def get_scorer(name):
    return {"lexicon": lexicon_scorer, "finbert": finbert_scorer}[name]


def assign_trading_date(published_utc, cal_dates, cutoff="15:30", tz="Asia/Kolkata"):
    """Map each publish time to the first trading date on which it could be acted on."""
    ts = pd.to_datetime(published_utc, utc=True)
    ist = ts.dt.tz_convert(tz)
    hh, mm = map(int, cutoff.split(":"))
    after_close = (ist.dt.hour * 60 + ist.dt.minute) >= hh * 60 + mm
    day = ist.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")

    cal = np.sort(pd.to_datetime(pd.Series(list(set(cal_dates)))).to_numpy(dtype="datetime64[ns]"))
    day_np = day.to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(cal, day_np, side="left")  # first trading date >= day
    valid = idx < len(cal)
    result = np.full(len(day_np), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    result[valid] = cal[idx[valid]]
    return pd.Series(result, index=published_utc.index)


def _empty_tkr():
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "ticker": pd.Series(dtype="object"),
            "tkr_sent": pd.Series(dtype="float64"),
            "tkr_count": pd.Series(dtype="float64"),
        }
    )


def aggregate(news, cal_dates, aliases, cutoff="15:30", tz="Asia/Kolkata"):
    """news needs columns published (UTC), title, sentiment. Returns (market_df, ticker_df)."""
    news = news.copy()
    news["date"] = assign_trading_date(news["published"], cal_dates, cutoff, tz)
    news = news.dropna(subset=["date"])
    news["date"] = news["date"].astype("datetime64[ns]")

    mkt = (
        news.groupby("date")
        .agg(mkt_sent=("sentiment", "mean"), mkt_count=("sentiment", "size"))
        .reset_index()
    )

    parts = []
    lower = news["title"].astype(str).str.lower()
    for ticker, kws in aliases.items():
        if not kws:
            continue
        pattern = "|".join(re.escape(k.lower()) for k in kws)
        sub = news[lower.str.contains(pattern, regex=True)]
        if sub.empty:
            continue
        g = (
            sub.groupby("date")
            .agg(tkr_sent=("sentiment", "mean"), tkr_count=("sentiment", "size"))
            .reset_index()
        )
        g["ticker"] = ticker
        parts.append(g[["date", "ticker", "tkr_sent", "tkr_count"]])
    tkr = pd.concat(parts, ignore_index=True) if parts else _empty_tkr()
    tkr["tkr_count"] = tkr["tkr_count"].astype("float64")
    return mkt, tkr
