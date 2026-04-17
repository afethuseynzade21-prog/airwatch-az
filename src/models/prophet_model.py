"""
AirWatch AZ — Prophet Model
=============================
Facebook Prophet for PM2.5 forecasting.

Why Prophet here:
  - Captures daily + yearly seasonality out-of-the-box
  - Robust to missing data and outliers
  - Generates uncertainty intervals (useful for business alerts)
  - Not strictly comparable to the feature-based models (uses only time axis),
    so treat its metrics as a separate decomposition benchmark.

Install: pip install prophet
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

log = logging.getLogger(__name__)


class _ProphetWrapper:
    """sklearn-style wrapper for Prophet so it fits the experiment interface."""

    def __init__(self, model):
        self._model = model

    def predict(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        future = pd.DataFrame({"ds": timestamps})
        fc     = self._model.predict(future)
        return fc["yhat"].clip(lower=0).values

    def predict_with_intervals(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        future = pd.DataFrame({"ds": timestamps})
        fc     = self._model.predict(future)
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "timestamp"})


def train_prophet(
    timestamps: pd.Series,
    y: pd.Series,
    tscv: TimeSeriesSplit,
    regressors: pd.DataFrame | None = None,
) -> tuple[dict, _ProphetWrapper | None]:
    """
    Train Prophet with optional external regressors (weather features).
    Uses TimeSeriesSplit for chronologically honest evaluation.

    Args:
        timestamps: DatetimeSeries aligned with y
        y:          PM2.5 target series
        tscv:       TimeSeriesSplit instance
        regressors: optional DataFrame of weather columns (same index as y)

    Returns:
        (result_dict, trained_wrapper_on_last_fold)
    """
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("Prophet not installed — skipping. Run: pip install prophet")
        return None, None

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    ts_arr = pd.to_datetime(timestamps).reset_index(drop=True)
    y_arr  = y.reset_index(drop=True)

    scores    = []
    last_wrap = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(y_arr)):
        df_train = pd.DataFrame({"ds": ts_arr.iloc[train_idx], "y": y_arr.iloc[train_idx]})
        df_test  = pd.DataFrame({"ds": ts_arr.iloc[test_idx],  "y": y_arr.iloc[test_idx]})

        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
            interval_width=0.9,
        )

        # Optionally add weather regressors
        if regressors is not None:
            reg_cols = ["temp", "humidity", "wind_speed", "precip"]
            reg_cols = [c for c in reg_cols if c in regressors.columns]
            for rc in reg_cols:
                m.add_regressor(rc)
            df_train = df_train.copy()
            df_test  = df_test.copy()
            for rc in reg_cols:
                df_train[rc] = regressors[rc].iloc[train_idx].values
                df_test[rc]  = regressors[rc].iloc[test_idx].values

        m.fit(df_train)
        fc = m.predict(df_test[["ds"] + ([rc for rc in (reg_cols if regressors is not None else [])])])

        y_true = df_test["y"].values
        y_pred = fc["yhat"].clip(lower=0).values[: len(y_true)]

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        mask = y_true > 1
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() else 0.0

        scores.append({"mae": mae, "rmse": rmse, "mape": mape, "r2": r2})
        last_wrap = _ProphetWrapper(m)
        log.info(f"  Prophet fold {fold+1}: MAE={mae:.3f}  R²={r2:.3f}")

    if not scores:
        return None, None

    result = {"model": "Prophet"}
    for m_key in ("mae", "rmse", "mape", "r2"):
        vals = [s[m_key] for s in scores]
        result[m_key]          = round(float(np.mean(vals)), 4)
        result[f"{m_key}_std"] = round(float(np.std(vals)), 4)

    log.info(f"Prophet → MAE={result['mae']:.3f}  R²={result['r2']:.3f}")
    return result, last_wrap
