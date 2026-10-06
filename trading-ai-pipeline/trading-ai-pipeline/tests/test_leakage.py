"""The tests that matter most: nothing from the future may reach a prediction."""
import numpy as np
import pandas as pd

from trading_ai.backtest import engine as E
from trading_ai.features import build as B
from trading_ai.features import news_features as NF
from trading_ai.ingestion import prices as P
from trading_ai.models import train as T


def _prices(n_days=400, tickers=("AAA", "BBB")):
    end = (pd.Timestamp("2020-01-01") + pd.offsets.BDay(n_days)).strftime("%Y-%m-%d")
    return P.synthetic_prices(list(tickers), "2020-01-01", end, seed=1)


def test_features_do_not_depend_on_future_prices():
    prices = _prices()
    full = B.build_dataset(prices)
    cutoff = np.sort(prices["date"].unique())[-60]
    trunc = B.build_dataset(prices[prices["date"] <= cutoff])
    feats = B.feature_columns(full)
    a = full[full["date"] <= cutoff].set_index(["date", "ticker"])[feats]
    b = trunc.set_index(["date", "ticker"])[feats]
    common = a.index.intersection(b.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_news_after_close_goes_to_next_trading_day():
    cal = pd.bdate_range("2024-01-01", "2024-01-12")  # Mon 1 Jan .. Fri 12 Jan
    # IST times -> UTC
    ist = lambda s: pd.Timestamp(s, tz="Asia/Kolkata").tz_convert("UTC")
    published = pd.Series(
        [
            ist("2024-01-02 10:00"),  # Tue morning   -> Tue
            ist("2024-01-02 16:00"),  # Tue after close -> Wed
            ist("2024-01-05 18:00"),  # Fri evening   -> Mon 8th
            ist("2024-01-06 11:00"),  # Saturday      -> Mon 8th
        ]
    )
    got = NF.assign_trading_date(published, cal).dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-08"]


def test_news_on_holiday_rolls_forward():
    cal = pd.DatetimeIndex(["2024-01-02", "2024-01-04"])  # Jan 3 is a "holiday"
    pub = pd.Series([pd.Timestamp("2024-01-03 05:00", tz="UTC")])  # 10:30 IST on Jan 3
    assert NF.assign_trading_date(pub, cal).iloc[0] == pd.Timestamp("2024-01-04")


def test_walk_forward_train_always_before_test():
    df = B.build_dataset(_prices(n_days=700))
    cfg = {"model": {"min_train_days": 300, "retrain_every_days": 21, "embargo_days": 1, "seed": 1}}
    preds = T.walk_forward(df, B.feature_columns(df), "logreg", cfg)
    assert len(preds) > 0
    assert (preds["fold_train_end"] < preds["date"]).all()
    # embargo respected: with embargo_days=1 there is >= 1 unused trading day between train end and first test day
    days = np.sort(df["date"].unique())
    pos = {d: i for i, d in enumerate(days)}
    for train_end, g in preds.groupby("fold_train_end"):
        assert pos[g["date"].min()] - pos[train_end] >= 2


def test_backtest_charges_costs_on_entry_only_for_buy_and_hold():
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = [{"date": d, "ticker": "X", "p_up": 1.0, "next_ret": 0.01} for d in dates]
    preds = pd.DataFrame(rows)
    daily = E.backtest(preds, threshold=0.5, cost_bps=10)
    assert np.isclose(daily.iloc[0], 0.01 - 0.0010)  # pays 10 bps once on entry
    assert np.allclose(daily.iloc[1:], 0.01)  # no further cost while holding


def test_backtest_stays_in_cash_when_not_confident():
    dates = pd.bdate_range("2024-01-01", periods=3)
    preds = pd.DataFrame({"date": dates, "ticker": "X", "p_up": 0.4, "next_ret": 0.05})
    assert np.allclose(E.backtest(preds), 0.0)


def test_fetch_prices_handles_yfinance_date_index(monkeypatch):
    """yfinance returns a tz-aware index named 'Date'; this used to create duplicate 'date' columns."""
    import sys
    import types

    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=5), name="Date").tz_localize("Asia/Kolkata")
    raw = pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100, "Dividends": 0.0, "Stock Splits": 0.0},
        index=idx,
    )
    fake = types.SimpleNamespace(Ticker=lambda t: types.SimpleNamespace(history=lambda **kw: raw.copy()))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    df = P.fetch_prices(["AAA.NS"], "2024-01-01")
    assert list(df.columns) == P.COLUMNS
    assert df["date"].is_unique and df["date"].dt.tz is None
