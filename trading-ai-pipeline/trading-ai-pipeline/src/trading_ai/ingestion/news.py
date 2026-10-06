"""News ingestion.

IMPORTANT: RSS feeds only return *recent* items, so there is no free history here.
Run this daily (the Airflow DAG does) and the archive grows over time. For a
long news backtest you need a historical dataset (e.g. a Kaggle financial-news
dump or GDELT) loaded into the same schema: published (UTC), title, source, link.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
NEWS_COLUMNS = ["published", "title", "source", "link"]

_POS = [
    "{n} shares surge after strong quarterly results",
    "{n} posts record profit, beats estimates",
    "Brokerages upgrade {n} on robust growth outlook",
    "{n} stock rallies as demand jumps",
]
_NEG = [
    "{n} shares fall after weak earnings",
    "{n} faces probe, stock slumps",
    "Brokerages downgrade {n} on margin concerns",
    "{n} warns of lower demand, shares drop",
]
_NEU = ["{n} announces board meeting date", "{n} to hold investor call next week", "Markets open flat, {n} in focus"]


def fetch_rss(feeds):
    import feedparser

    rows = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
        except Exception as ex:
            log.warning("RSS failed for %s: %s", url, ex)
            continue
        for e in parsed.entries:
            pp = e.get("published_parsed") or e.get("updated_parsed")
            if not pp:
                continue
            rows.append(
                {
                    "published": pd.Timestamp(*pp[:6], tz="UTC"),
                    "title": e.get("title", "").strip(),
                    "source": url,
                    "link": e.get("link", ""),
                }
            )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def synthetic_news(start, end, aliases, seed=42, tz="Asia/Kolkata"):
    """Random headlines with NO relationship to returns (so any 'edge' found is a bug)."""
    rng = np.random.default_rng(seed + 1)
    names = [v[0].title() for v in aliases.values() if v] or ["Sensex"]
    rows = []
    for d in pd.bdate_range(start, end):
        for _ in range(rng.integers(0, 5)):
            kind = rng.choice(["pos", "neg", "neu"])
            tpl = rng.choice({"pos": _POS, "neg": _NEG, "neu": _NEU}[kind])
            minutes = int(rng.integers(6 * 60, 21 * 60))
            ts = (pd.Timestamp(d).tz_localize(tz) + pd.Timedelta(minutes=minutes)).tz_convert("UTC")
            rows.append(
                {"published": ts, "title": tpl.format(n=rng.choice(names)), "source": "synthetic", "link": ""}
            )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def save_raw(df, folder: Path):
    if df.empty:
        log.warning("No news fetched; nothing saved.")
        return None
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = folder / f"news_{stamp}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_all(folder: Path):
    """Concatenate every snapshot, dedupe. Returns an empty frame if nothing collected yet."""
    files = sorted(Path(folder).glob("news_*.parquet"))
    if not files:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.drop_duplicates(subset=["title", "published"]).sort_values("published")
    return df.reset_index(drop=True)
