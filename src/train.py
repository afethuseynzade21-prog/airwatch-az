"""
AirWatch AZ — Model Training Orchestrator
==========================================
Runs a 5-model experiment with TimeSeriesSplit and SHAP analysis.

Models compared:
  1. Persistence Baseline     — trivial floor
  2. Ridge Regression         — linear reference
  3. Random Forest            — non-linear tabular
  4. LightGBM                 — gradient boosting (primary)
  5. LSTM (PyTorch)           — sequence-aware deep model
  6. Prophet                  — decomposition-based (optional)

Usage:
    python -m src.train
    # Or programmatically:
    from src.train import run_experiment
    results, best_model = run_experiment(X, y, timestamps)
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import MODEL_DIR
from src.models.baseline import persistence_baseline, train_ridge
from src.models.tree_models import train_random_forest, train_lightgbm
from src.models.lstm import train_lstm
from src.models.prophet_model import train_prophet

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SHAP analysis
# ════════════════════════════════════════════════════════════════════════════

def compute_shap(model, X: pd.DataFrame, model_name: str, max_rows: int = 500) -> pd.DataFrame | None:
    """
    Compute SHAP values for tree-based models.
    Returns a DataFrame of mean absolute SHAP per feature (sorted descending).
    """
    try:
        import shap
    except ImportError:
        log.warning("SHAP not installed — skipping. Run: pip install shap")
        return None

    if model_name not in ("RandomForest", "LightGBM"):
        return None

    X_sample = X.sample(min(max_rows, len(X)), random_state=42)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        df_shap = pd.DataFrame({
            "feature":     X_sample.columns,
            "shap_abs_mean": np.abs(shap_vals).mean(axis=0),
        }).sort_values("shap_abs_mean", ascending=False).reset_index(drop=True)
        log.info(f"SHAP computed for {model_name} ({len(X_sample)} samples)")
        return df_shap
    except Exception as exc:
        log.warning(f"SHAP computation failed: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Error analysis
# ════════════════════════════════════════════════════════════════════════════

def error_analysis(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """
    Compute prediction errors segmented by hour, weekday, and season.
    Reveals systematic bias patterns (e.g. rush-hour underestimation).
    """
    try:
        if model_name == "LSTM":
            preds = model.predict(X.values)
            # LSTM drops seq_len rows from the front
            seq_len = model.seq_len
            y_aligned = y.values[seq_len:]
            ts_aligned = pd.to_datetime(timestamps).values[seq_len:]
        else:
            preds = model.predict(X)
            y_aligned = y.values
            ts_aligned = pd.to_datetime(timestamps).values

        n = min(len(preds), len(y_aligned))
        df_err = pd.DataFrame({
            "timestamp": ts_aligned[:n],
            "actual":    y_aligned[:n],
            "predicted": preds[:n],
        })
        df_err["error"]    = df_err["predicted"] - df_err["actual"]
        df_err["abs_error"]= df_err["error"].abs()
        df_err["hour"]     = pd.DatetimeIndex(df_err["timestamp"]).hour
        df_err["weekday"]  = pd.DatetimeIndex(df_err["timestamp"]).dayofweek
        df_err["month"]    = pd.DatetimeIndex(df_err["timestamp"]).month
        df_err["season"]   = df_err["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3:  "Spring", 4: "Spring", 5: "Spring",
            6:  "Summer", 7: "Summer", 8: "Summer",
            9:  "Autumn",10: "Autumn",11: "Autumn",
        })
        return df_err
    except Exception as exc:
        log.warning(f"Error analysis failed: {exc}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# Main experiment
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series | None = None,
    n_splits: int = 5,
    save_model: bool = True,
    run_lstm: bool = True,
    run_prophet: bool = False,
) -> tuple[pd.DataFrame, object, dict]:
    """
    Run the full 5-model experiment.

    Args:
        X:           Feature matrix (from build_features)
        y:           PM2.5 target series
        timestamps:  DatetimeSeries aligned with X/y (for Prophet + error analysis)
        n_splits:    TimeSeriesSplit folds
        save_model:  Persist best model + results to outputs/
        run_lstm:    Include LSTM (slow; requires torch)
        run_prophet: Include Prophet (requires prophet package)

    Returns:
        (results_df, best_model, artifacts)
        artifacts: {shap_df, error_df}
    """
    log.info("=" * 60)
    log.info("AirWatch AZ — Model Experiment")
    log.info(f"Dataset: {len(X):,} rows × {X.shape[1]} features")
    log.info(f"TimeSeriesSplit: {n_splits} folds")
    log.info("=" * 60)

    tscv    = TimeSeriesSplit(n_splits=n_splits)
    results = []
    models  = {}

    # 1. Persistence
    log.info("\n[1/5] Persistence Baseline...")
    results.append(persistence_baseline(X, y, tscv))

    # 2. Ridge
    log.info("\n[2/5] Ridge Regression...")
    scores_ridge, model_ridge = train_ridge(X, y, tscv)
    results.append(scores_ridge)
    models["Ridge"] = model_ridge

    # 3. Random Forest
    log.info("\n[3/5] Random Forest...")
    scores_rf, model_rf = train_random_forest(X, y, tscv)
    results.append(scores_rf)
    models["RandomForest"] = model_rf

    # 4. LightGBM
    log.info("\n[4/5] LightGBM...")
    scores_lgb, model_lgb = train_lightgbm(X, y, tscv)
    if scores_lgb:
        results.append(scores_lgb)
        models["LightGBM"] = model_lgb

    # 5. LSTM
    if run_lstm:
        log.info("\n[5/5] LSTM (PyTorch)...")
        scores_lstm, model_lstm = train_lstm(X, y, tscv)
        if scores_lstm:
            results.append(scores_lstm)
            models["LSTM"] = model_lstm

    # 6. Prophet (optional)
    if run_prophet and timestamps is not None:
        log.info("\n[+] Prophet...")
        weather_cols = [c for c in ["temp", "humidity", "wind_speed", "precip"] if c in X.columns]
        regressors = X[weather_cols] if weather_cols else None
        scores_ph, model_ph = train_prophet(timestamps, y, tscv, regressors)
        if scores_ph:
            results.append(scores_ph)
            models["Prophet"] = model_ph

    # ── Leaderboard ──────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results).set_index("model").sort_values("mae")

    print("\n" + "=" * 70)
    print("  AIRWATCH AZ — MODEL LEADERBOARD  (TimeSeriesSplit, n={})".format(n_splits))
    print("=" * 70)
    display_cols = ["mae", "mae_std", "rmse", "rmse_std", "mape", "r2"]
    display_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[display_cols].round(3).to_string())
    print("=" * 70)
    print("  MAE / RMSE in μg/m³  |  MAPE in %  |  R²: higher is better")
    print("=" * 70 + "\n")

    # ── Best model selection (by MAE, excluding baselines) ───────────────────
    candidate_order = ["LightGBM", "LSTM", "RandomForest", "Ridge"]
    best_name  = next((n for n in candidate_order if n in models), None)
    best_model = models.get(best_name) if best_name else None

    if best_name:
        log.info(f"Best model: {best_name} (MAE={df_results.loc[best_name, 'mae']:.3f})")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    shap_df = None
    if best_name and best_model and best_name in ("LightGBM", "RandomForest"):
        log.info("Computing SHAP feature importance...")
        shap_df = compute_shap(best_model, X, best_name)
        if shap_df is not None:
            print("\nTop 10 Features (SHAP):")
            print(shap_df.head(10).to_string(index=False))

    # ── Error analysis ────────────────────────────────────────────────────────
    error_df = pd.DataFrame()
    if best_model and timestamps is not None:
        log.info("Running error analysis...")
        error_df = error_analysis(best_model, X, y, timestamps, best_name or "")

        if not error_df.empty:
            hourly_mae = error_df.groupby("hour")["abs_error"].mean()
            worst_hour = int(hourly_mae.idxmax())
            best_hour  = int(hourly_mae.idxmin())
            log.info(f"Error by hour: worst={worst_hour:02d}:00 ({hourly_mae[worst_hour]:.2f} μg/m³), "
                     f"best={best_hour:02d}:00 ({hourly_mae[best_hour]:.2f} μg/m³)")
            seasonal_mae = error_df.groupby("season")["abs_error"].mean().sort_values(ascending=False)
            log.info(f"Error by season:\n{seasonal_mae.round(2)}")

    # ── Persist artifacts ─────────────────────────────────────────────────────
    if save_model and best_model:
        model_path = MODEL_DIR / "best_model.pkl"
        joblib.dump({
            "model":      best_model,
            "name":       best_name,
            "features":   list(X.columns),
            "n_samples":  len(X),
            "n_features": X.shape[1],
        }, model_path)
        log.info(f"Model saved: {model_path}")

        # Results JSON
        results_path = MODEL_DIR / "results.json"
        results_path.write_text(
            json.dumps(df_results.reset_index().to_dict(orient="records"), indent=2)
        )

        # SHAP CSV
        if shap_df is not None:
            shap_df.to_csv(MODEL_DIR / "shap_importance.csv", index=False)

        # Error analysis CSV
        if not error_df.empty:
            error_df.to_csv(MODEL_DIR / "error_analysis.csv", index=False)

        log.info(f"Artifacts saved to {MODEL_DIR}/")

    return df_results, best_model, {"shap": shap_df, "errors": error_df}


def load_model(path: str = "outputs/best_model.pkl") -> tuple[object, str, list]:
    data = joblib.load(path)
    return data["model"], data["name"], data["features"]


if __name__ == "__main__":
    from src.data_pipeline import fetch_all
    from src.features import build_features

    df = fetch_all(days=365)
    X, y, ts = build_features(df)
    results, best, artifacts = run_experiment(X, y, ts, n_splits=5)
