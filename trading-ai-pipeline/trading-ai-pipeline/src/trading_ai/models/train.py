"""Walk-forward (expanding window) training. No random splits, ever."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name, seed=7):
    if name == "logreg":
        return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=seed
        )
    raise ValueError(f"Unknown model: {name}")


def walk_forward(df, feature_cols, model_name, cfg):
    """Train on the past, predict the next block, slide forward. Returns out-of-sample predictions only."""
    mcfg = cfg["model"]
    data = df.dropna(subset=["target"]).sort_values(["date", "ticker"])
    dates = np.sort(data["date"].unique())
    min_train, step, embargo = mcfg["min_train_days"], mcfg["retrain_every_days"], mcfg["embargo_days"]
    if len(dates) <= min_train + embargo:
        raise ValueError(f"Not enough history: {len(dates)} days, need > {min_train + embargo}")

    out = []
    for start in range(min_train + embargo, len(dates), step):
        train_end = dates[start - embargo - 1]
        test_dates = dates[start : start + step]
        train = data[data["date"] <= train_end]
        test = data[data["date"].isin(test_dates)]
        model = make_model(model_name, mcfg["seed"]).fit(train[feature_cols], train["target"])
        res = test[["date", "ticker", "ret_1", "next_ret", "target"]].copy()
        res["p_up"] = model.predict_proba(test[feature_cols])[:, 1]
        res["fold_train_end"] = pd.Timestamp(train_end)
        out.append(res)
    return pd.concat(out, ignore_index=True)
