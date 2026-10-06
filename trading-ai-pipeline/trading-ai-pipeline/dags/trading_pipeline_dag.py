"""Airflow DAG: runs every weekday after NSE close (15:30 IST).

    ingest_prices ─┐
                   ├─> build_features ─> train_and_backtest ─> paper_trade
    ingest_news  ──┘

Setup: install Airflow separately (follow its constraints file), then point
AIRFLOW__CORE__DAGS_FOLDER at this dags/ directory.
"""
import pathlib
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trading_ai import pipeline as p  # noqa: E402

default_args = {"owner": "pranoy", "retries": 2, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="trading_ai_daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Kolkata"),
    schedule="0 17 * * 1-5",  # 17:00 IST, Mon-Fri
    catchup=False,
    default_args=default_args,
    tags=["trading", "portfolio"],
) as dag:
    prices = PythonOperator(task_id="ingest_prices", python_callable=p.ingest_prices)
    news = PythonOperator(task_id="ingest_news", python_callable=p.ingest_news)
    features = PythonOperator(task_id="build_features", python_callable=p.build_features)
    train = PythonOperator(task_id="train_and_backtest", python_callable=p.train_and_backtest)
    paper = PythonOperator(task_id="paper_trade", python_callable=p.paper_trade)

    [prices, news] >> features >> train >> paper
