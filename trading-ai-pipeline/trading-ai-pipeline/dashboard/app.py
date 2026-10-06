"""AI Stock Coach: a beginner-friendly front end for the trading pipeline.

Run:  streamlit run dashboard/app.py
Demo with fake data:  streamlit run dashboard/app.py -- --synthetic
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trading_ai import pipeline  # noqa: E402
from trading_ai.config import load_config  # noqa: E402

SYNTHETIC = "--synthetic" in sys.argv or os.environ.get("TRADING_AI_SYNTHETIC") == "1"
cfg = load_config(synthetic=SYNTHETIC)
proc, paper_dir = cfg["paths"]["processed"], cfg["paths"]["paper"]

STOCK_NAMES = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
}
STRATEGY_NAMES = {
    "buy_hold": "Just buy and hold",
    "logreg": "AI (simple model)",
    "hgb": "AI (advanced model)",
    "prev_day": "Copy yesterday's move",
}

st.set_page_config(page_title="AI Stock Coach", page_icon="📈", layout="wide")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("📈 AI Stock Coach")
    st.caption("Learn how AI reads the market, using practice money only.")
    st.write("**Daily routine:** after 3:30 PM IST (market close), click the button below.")
    if st.button("Update with latest market data", type="primary", width="stretch"):
        with st.spinner("Downloading prices and updating the AI. This takes a minute or two..."):
            try:
                pipeline.run_all(cfg)
                st.success("Updated!")
            except Exception as ex:  # network, data issues
                st.error(f"Update failed: {ex}")
    st.divider()
    st.caption("Educational tool. Not investment advice. Not SEBI-registered. Practice money only.")

st.title("AI Stock Coach")
st.write("See what an AI model thinks about a few big Indian stocks, how good it really is, and learn the basics along the way.")
if SYNTHETIC:
    st.warning("Demo mode: this is FAKE data used to test the app. Don't read anything into the numbers.")

metrics_path = proc / "metrics.json"
dataset_path = proc / "dataset.parquet"
if not (metrics_path.exists() and dataset_path.exists()):
    st.info("Welcome! There's no data yet. Click **Update with latest market data** in the sidebar to get started.")
    st.stop()

metrics = pd.DataFrame(json.loads(metrics_path.read_text())).set_index("model")
equity = pd.read_csv(proc / "equity_curves.csv", parse_dates=["date"]).set_index("date")
dataset = pd.read_parquet(dataset_path)
log_path = paper_dir / "predictions_log.csv"
log = pd.read_csv(log_path, parse_dates=["pred_date"]) if log_path.exists() else pd.DataFrame()

tab_today, tab_score, tab_record, tab_learn = st.tabs(
    ["Today's view", "How good is the AI?", "My practice record", "Learn the basics"]
)

# ---------------- Today ----------------
with tab_today:
    last_date = dataset["date"].max()
    st.subheader(f"The AI's view after the close on {last_date:%d %b %Y}")
    st.caption("These are guesses about the NEXT trading day. The AI only ever chooses between 'hold the stock' and 'stay in cash'.")

    latest = dataset[dataset["date"] == last_date].set_index("ticker")
    today = pd.DataFrame()
    if not log.empty:
        today = log[(log["pred_date"] == last_date) & (log["model"] == cfg["paper"]["model"])]
    if today.empty:
        st.info("No predictions logged yet. Click **Update** in the sidebar.")
    else:
        today = today.sort_values("p_up", ascending=False)
        cols = st.columns(3)
        for i, (_, row) in enumerate(today.iterrows()):
            p = float(row["p_up"])
            gap = abs(p - 0.5)
            confidence = "Very unsure" if gap < 0.03 else "Slightly leaning" if gap < 0.06 else "Fairly confident"
            action = "🟢 Hold it tomorrow" if p > 0.5 else "⚪ Stay in cash"
            info = latest.loc[row["ticker"]] if row["ticker"] in latest.index else None
            with cols[i % 3].container(border=True):
                st.subheader(STOCK_NAMES.get(row["ticker"], row["ticker"]))
                st.caption(row["ticker"])
                if info is not None:
                    st.metric("Last price", f"₹{info['close']:,.2f}", f"{info['ret_1'] * 100:+.2f}% today")
                st.write(f"**AI says: {action}**")
                st.progress(min(max(p, 0.0), 1.0))
                st.caption(f"Chance of going up tomorrow: **{p:.0%}**. {confidence}.")
        if (today["p_up"].between(0.45, 0.55)).all():
            st.info(
                "Notice that every stock is close to 50%. That means the AI is basically unsure about all of them. "
                "That's normal: predicting tomorrow's move is very hard, even for professionals."
            )

# ---------------- Scoreboard ----------------
with tab_score:
    st.subheader("If you had followed the AI in the past...")
    st.caption(
        f"A fair test: the AI only predicted days it had never seen. Trading costs included. "
        f"Period: {equity.index.min():%b %Y} to {equity.index.max():%b %Y}."
    )
    amount = st.number_input("Pretend you started with (₹)", min_value=1000, value=100000, step=10000)

    final = equity.iloc[-1] * amount
    rows = []
    for key in equity.columns:
        m = metrics.loc[key]
        rows.append(
            {
                "Strategy": STRATEGY_NAMES.get(key, key),
                "Your money would now be": f"₹{final[key]:,.0f}",
                "Gain / loss": f"{m['total_return']:+.1%}",
                "Worst fall along the way": f"{m['max_drawdown']:.0%}",
                "Right about up/down": f"{m['accuracy']:.1%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.line_chart(equity.rename(columns=STRATEGY_NAMES) * amount)

    ai_keys = [k for k in ("logreg", "hgb") if k in metrics.index]
    best = metrics.loc[ai_keys, "total_return"].idxmax()
    hold_ret = metrics.loc["buy_hold", "total_return"]
    best_ret = metrics.loc[best, "total_return"]
    lo, hi, acc = metrics.loc[best, ["acc_ci95_low", "acc_ci95_high", "accuracy"]]

    st.subheader("The bottom line")
    if best_ret < hold_ret:
        st.error(
            f"Just holding the stocks ({hold_ret:+.0%}) did better than the best AI "
            f"({STRATEGY_NAMES[best]}: {best_ret:+.0%})."
        )
    else:
        st.success(
            f"The best AI ({STRATEGY_NAMES[best]}: {best_ret:+.0%}) beat holding ({hold_ret:+.0%}) in this test. "
            "One test isn't proof, so keep watching the live record."
        )
    if lo <= 0.5 <= hi:
        st.warning(
            f"The AI was right {acc:.1%} of the time. That's within the range of a coin flip "
            f"({lo:.1%} to {hi:.1%}), so there's **no evidence it has a real edge yet**."
        )
    else:
        st.info(f"The AI was right {acc:.1%} of the time, statistically different from a coin flip. Promising, but verify it live.")

# ---------------- Practice record ----------------
with tab_record:
    st.subheader("Live scorecard: the AI's real predictions, checked against what happened")
    st.write(
        "Every day you click **Update**, the AI logs its guesses. The next day they're marked right or wrong. "
        "This is the most honest test, because the AI can't cheat on the future."
    )
    if log.empty:
        st.info("Nothing logged yet. Click **Update** in the sidebar after the market closes.")
    else:
        settled = log[log["status"] == "settled"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions made", len(log))
        c2.metric("Already checked", len(settled))
        c3.metric("Live hit rate", f"{settled['correct'].mean():.0%}" if len(settled) else "not yet")
        st.caption("Be patient: you need 100+ checked predictions before a hit rate means anything. Small samples are just luck.")

        view = log.sort_values("pred_date", ascending=False).head(30).copy()
        view["Date"] = view["pred_date"].dt.strftime("%d %b %Y")
        view["Stock"] = view["ticker"].map(STOCK_NAMES).fillna(view["ticker"])
        view["AI's chance of going up"] = (view["p_up"] * 100).round(0).astype(int).astype(str) + "%"
        view["AI said"] = np.where(view["p_up"] > 0.5, "Hold", "Cash")
        view["Price moved"] = view["actual_ret"].map(lambda x: "waiting..." if pd.isna(x) else f"{x:+.2%}")
        view["Right?"] = view["correct"].map({1.0: "✅", 0.0: "❌"}).fillna("⏳")
        st.dataframe(
            view[["Date", "Stock", "AI's chance of going up", "AI said", "Price moved", "Right?"]],
            hide_index=True,
            width="stretch",
        )

# ---------------- Learn ----------------
with tab_learn:
    st.subheader("Trading basics in plain words")
    topics = {
        "What does 'buy' and 'sell' mean?": "Buying a stock means owning a tiny piece of a company. If its price rises you profit, if it falls you lose. Selling closes the position and locks in the result.",
        "What is 'hold' vs 'stay in cash'?": "This AI only makes one decision per stock per day: hold it (it expects a rise) or stay in cash (it doesn't). It never bets on prices falling.",
        "What does '55% chance of going up' mean?": "If the AI said 55% on 100 similar days, it expects the price to rise on about 55 of them. 50% is a coin flip, so anything near 50% means the AI doesn't know.",
        "What is 'buy and hold'?": "Buying and never selling. It sounds too simple to beat, but it's the benchmark: if an AI can't beat it, you're better off doing nothing.",
        "What is 'worst fall along the way'?": "The biggest drop from a high point to a later low point (called max drawdown). A -30% means that at some moment you'd have been down 30% from your best.",
        "Why do trading costs matter so much?": "Every buy and sell costs a small fee and tax. A strategy that trades every day pays them again and again, which can wipe out a tiny edge.",
        "Why is it so hard for AI to predict stocks?": "Prices already reflect what everyone knows, news is priced in within minutes, and daily moves are mostly random noise. Even tiny real edges are rare and get competed away.",
        "How do I practice safely?": "Use a paper trading (virtual money) mode. TradingView has one, and many broker apps offer a demo mode. Don't use binary-options apps. Never trade money you can't afford to lose.",
        "Is this investment advice?": "No. This is an educational project. The AI has not been shown to beat simply holding stocks, so please don't use it to trade real money.",
    }
    for q, a in topics.items():
        with st.expander(q):
            st.write(a)
