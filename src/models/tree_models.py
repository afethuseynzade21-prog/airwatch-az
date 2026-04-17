"""
Tree-based models: Random Forest and LightGBM.
These are the production workhorses for tabular time-series.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.config import LGB_PARAMS, RF_PARAMS

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


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    tscv: TimeSeriesSplit,
) -> tuple[dict, RandomForestRegressor]:
    """
    Random Forest with TimeSeriesSplit cross-validation.
    Strong non-linear baseline; slower than LightGBM but robust.
    """
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr, y_tr)
        s = _metrics(y_te, m.predict(X_te))
        scores.append(s)
        log.info(f"  RF fold {fold+1}: MAE={s['mae']:.3f}  R²={s['r2']:.3f}")

    model_final = RandomForestRegressor(**RF_PARAMS).fit(X, y)
    result = _aggregate(scores, "RandomForest")
    log.info(f"RandomForest → MAE={result['mae']:.3f}  R²={result['r2']:.3f}")
    return result, model_final


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    tscv: TimeSeriesSplit,
) -> tuple[dict, object] | tuple[None, None]:
    """
    LightGBM with early stopping on each fold.
    Best balance of speed and accuracy for this dataset size.
    Install: pip install lightgbm
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LightGBM not installed — skipping. Run: pip install lightgbm")
        return None, None

    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        s = _metrics(y_te, m.predict(X_te))
        scores.append(s)
        log.info(f"  LGB fold {fold+1}: MAE={s['mae']:.3f}  R²={s['r2']:.3f}  trees={m.best_iteration_}")

    model_final = lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y)
    result = _aggregate(scores, "LightGBM")
    log.info(f"LightGBM → MAE={result['mae']:.3f}  R²={result['r2']:.3f}")
    return result, model_final
