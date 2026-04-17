"""
Baseline models: Persistence and Ridge Regression.
Persistence is the minimum bar — every other model must beat it.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mask = y_true > 1
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() else 0.0
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def _aggregate(scores: list[dict], name: str) -> dict:
    result = {"model": name}
    for m in ("mae", "rmse", "mape", "r2"):
        vals = [s[m] for s in scores]
        result[m]          = round(float(np.mean(vals)), 4)
        result[f"{m}_std"] = round(float(np.std(vals)), 4)
    return result


def persistence_baseline(X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> dict:
    """
    Predict t+1 = value at t (last observed PM2.5).
    The simplest possible forecaster — any real model must beat this.
    """
    scores = []
    for _, test_idx in tscv.split(X):
        y_test = y.iloc[test_idx].values
        y_pred = X["pm25_lag_1h"].iloc[test_idx].values
        scores.append(_metrics(y_test, y_pred))
    result = _aggregate(scores, "Persistence")
    log.info(f"Persistence → MAE={result['mae']:.3f}  R²={result['r2']:.3f}")
    return result


def train_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    tscv: TimeSeriesSplit,
) -> tuple[dict, object]:
    """
    Ridge regression with StandardScaler.
    Interpretable, fast, and useful as a linear baseline.
    """
    scores = []
    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        m = Ridge(alpha=10.0)
        m.fit(X_tr_sc, y_tr)
        scores.append(_metrics(y_te, m.predict(X_te_sc)))
        log.info(f"  Ridge fold {fold+1}: MAE={scores[-1]['mae']:.3f}")

    scaler_final = StandardScaler().fit(X)
    model_final  = Ridge(alpha=10.0).fit(scaler_final.transform(X), y)
    # Attach scaler for inference
    model_final._scaler = scaler_final  # type: ignore[attr-defined]

    result = _aggregate(scores, "Ridge")
    log.info(f"Ridge → MAE={result['mae']:.3f}  R²={result['r2']:.3f}")
    return result, model_final
