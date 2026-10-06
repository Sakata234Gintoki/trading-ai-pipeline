# Trading AI Pipeline

End-to-end data + ML project: ingest prices and news, build leak-free features, train with walk-forward validation, backtest with costs, and paper trade with a prediction log. **Research / portfolio project. Paper money only. Not financial advice.**

## Workflow

```
 yfinance (prices) ─┐                                        ┌─> equity curves + metrics.json
                    ├─> data/raw (append-only snapshots)     │
 RSS feeds (news) ──┘          │                             │
                               v                             │
                    features: returns, vol, RSI, MACD,       │
                    news sentiment (cutoff-aware)            │
                               │                             │
                               v                             │
            walk-forward training (logreg, hist-GBM) ────> backtest (costs in) ──> Streamlit dashboard
                               │
                               v
                  paper trading log (predict -> wait -> settle -> live accuracy)
```

Airflow runs it every weekday at 17:00 IST (after NSE close).

## Folder structure

```
trading-ai-pipeline/
├── config/config.yaml           # tickers, feeds, model + backtest settings
├── src/trading_ai/
│   ├── ingestion/prices.py      # yfinance + synthetic generator
│   ├── ingestion/news.py        # RSS + synthetic generator
│   ├── features/build.py        # price features + target
│   ├── features/news_features.py# sentiment + the "after-close news -> next day" rule
│   ├── models/train.py          # walk-forward training
│   ├── backtest/engine.py       # long-or-cash backtest, costs, metrics, baselines
│   ├── paper.py                 # prediction log + settlement
│   ├── pipeline.py              # stage functions (used by CLI and Airflow)
│   └── cli.py                   # python -m trading_ai.cli <stage>
├── dags/trading_pipeline_dag.py # Airflow DAG
├── dashboard/app.py             # Streamlit monitor
├── tests/test_leakage.py        # leakage + backtest correctness tests
├── data/{raw,processed,paper}/  # created at runtime
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .      # makes `trading_ai` importable, no PYTHONPATH needed

make test         # 6 leakage/backtest tests, should all pass
make synthetic    # offline smoke test on fake data (no internet needed)
make run          # real data: yfinance + RSS (needs internet)
make dashboard    # AI Stock Coach: beginner-friendly app (click 'Update' in the sidebar each day after 3:30 PM IST)
```

Without `make`: `PYTHONPATH=src python -m trading_ai.cli all` (add `--synthetic` for fake data). Stages: `ingest`, `features`, `train`, `paper`, `all`.

## Build order (do it in this order)

1. **Price-only first.** In `config.yaml` set `features.use_news: false`, run `make run`. Get the whole loop working end to end.
2. **Read the results honestly.** Compare each model to `buy_hold` and `prev_day`. If the accuracy CI includes 50% and nothing beats buy-and-hold after costs, there is no proven edge. That's a normal result.
3. **Start collecting news daily** (Airflow does this, or run `cli ingest` yourself). RSS gives only recent items, so your news archive starts the day you start. For a long news backtest, load a historical dataset (Kaggle financial news, GDELT) into the same schema: `published` (UTC), `title`, `source`, `link`.
4. **Turn news on** and see if it adds anything over price-only. Try `sentiment: finbert` (`pip install transformers torch`).
5. **Paper trade for months.** The log in `data/paper/predictions_log.csv` is the real test: live, timestamped, can't be overfit.

## Rules this project enforces

- Features at day *t* use only data up to the close of *t* (tested: change the future, features don't move).
- News published at/after 15:30 IST only counts for the next trading day; weekends and holidays roll forward.
- Walk-forward splits only, with an embargo gap. No random train/test splits.
- Costs (default 10 bps per side) are in every backtest. Idle capital earns 0.
- Every model is compared to buy-and-hold and "yesterday's direction".

## Known limits (be upfront about these in interviews)

- Daily direction prediction on public data is close to a coin flip; a flat result is expected.
- Yahoo data is free but not institutional quality (adjustments, gaps, survivorship bias in ticker choice).
- Lexicon sentiment is crude. FinBERT is better but still headline-level.
- Backtest is long-or-cash only (no shorting), equal weight, 5 large caps, so results aren't general.

## Ideas for phase 2

- dbt models + Postgres for the clean layer (raw Parquet -> staging -> marts).
- More tickers / sectors; index and macro features (USDINR, crude, India VIX).
- Probability calibration and a minimum-edge threshold instead of 0.5.
- Drift monitoring and data-freshness checks as Airflow tasks.
- Experiment tracking with MLflow.
