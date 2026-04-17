"""
AirWatch AZ — Real-Time Inference Pipeline
==========================================
Generates hourly PM2.5 forecasts and persists them to DuckDB.

Architecture:
  1. Load latest readings from DB (or fetch live)
  2. Build features on the rolling window
  3. Run best model → predictions for next H hours
  4. Persist predictions to forecasts table
  5. Return structured result with risk classification

Design choice: the pipeline is stateless — it re-reads from
the DB each call, so it can be safely called from cron, the
API, or the dashboard without shared-state race conditions.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config import WHO_THRESHOLDS, MODEL_DIR
from src.features import build_features

log = logging.getLogger(__name__)


# ── Risk classification ───────────────────────────────────────────────────────

def classify_risk(pm25: float) -> dict:
    """Map a PM2.5 value to a WHO risk level with action text."""
    for t in WHO_THRESHOLDS:
        if t["min"] <= pm25 < t["max"]:
            return {
                "label":  t["label"],
                "risk":   t["risk"],
                "color":  t["color"],
                "action": t["action"],
                "pm25":   round(pm25, 1),
                "who_ratio": round(pm25 / 5.0, 1),   # WHO annual guideline = 5 μg/m³
            }
    return {**WHO_THRESHOLDS[-1], "pm25": round(pm25, 1), "who_ratio": round(pm25 / 5.0, 1)}


def classify_risk_series(pm25_series: pd.Series) -> pd.DataFrame:
    """Vectorised risk classification for a forecast window."""
    rows = [classify_risk(v) for v in pm25_series]
    return pd.DataFrame(rows)


# ── Health recommendations (business layer) ───────────────────────────────────

_HEALTH_MESSAGES = {
    "low":      "✅ Air quality is good. Safe for all outdoor activities.",
    "medium":   "⚠️ Sensitive groups (asthma, elderly, children) should limit prolonged outdoor exertion.",
    "high":     "🚨 Reduce outdoor exercise. Wear N95 mask outdoors. Close windows.",
    "critical": "🔴 Avoid all outdoor activity. Vulnerable individuals should stay indoors.",
    "extreme":  "☣️ Emergency conditions. Evacuate sensitive populations. Contact local authorities.",
}

_POLICY_MESSAGES = {
    "low":      "No policy action required.",
    "medium":   "Issue public advisory for schools and healthcare facilities.",
    "high":     "Activate Stage 1 protocol: restrict industrial emissions, increase public transport.",
    "critical": "Activate Stage 2 protocol: mandatory traffic restrictions, industrial shutdown orders.",
    "extreme":  "Activate Emergency protocol: coordinate with national health authorities immediately.",
}


def get_recommendations(risk_level: str) -> dict:
    return {
        "health": _HEALTH_MESSAGES.get(risk_level, ""),
        "policy": _POLICY_MESSAGES.get(risk_level, ""),
    }


# ── Feature building for inference ───────────────────────────────────────────

def _build_inference_window(df_recent: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build features on the most recent data window.
    Returns (X_last_row, feature_columns) for single-step prediction.
    """
    X, y, ts = build_features(df_recent)
    if X.empty:
        raise ValueError("Insufficient data to build inference features")
    return X.iloc[[-1]], list(X.columns)   # last row = current state


# ── Multi-step forecast ───────────────────────────────────────────────────────

def forecast_horizon(
    model,
    model_name: str,
    feature_names: list[str],
    df_history: pd.DataFrame,
    horizon_h: int = 24,
) -> pd.DataFrame:
    """
    Generate a multi-step forecast by iterating the model.

    For each future step:
      1. Append last prediction to the rolling window
      2. Rebuild lag/rolling features
      3. Predict next step

    This is the "recursive multi-step" strategy — simple and
    model-agnostic, at the cost of error accumulation.

    Returns DataFrame: timestamp, pm25_pred, pm25_lower, pm25_upper, risk, ...
    """
    df = df_history.copy().sort_values("timestamp").reset_index(drop=True)
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])

    predictions = []
    noise_std = df["pm25"].std() * 0.15   # uncertainty grows with horizon

    for h in range(1, horizon_h + 1):
        target_ts = last_ts + timedelta(hours=h)
        try:
            X_now, _ = _build_inference_window(df)
            # Align features to trained model's feature list
            missing = [f for f in feature_names if f not in X_now.columns]
            for mf in missing:
                X_now[mf] = 0.0
            X_now = X_now[feature_names]

            if model_name == "LSTM":
                # LSTM needs a full sequence window; use raw feature matrix
                X_full, _, _ = build_features(df)
                if len(X_full) < model.seq_len:
                    break
                pm25_pred = float(model.predict(X_full.values[-model.seq_len - 1 :])[-1])
            else:
                pm25_pred = float(model.predict(X_now)[0])

            pm25_pred = max(0.0, pm25_pred)
            # Uncertainty interval: ±1.5σ × √h (random walk uncertainty growth)
            uncertainty = noise_std * (h ** 0.5)
            pm25_lower  = max(0.0, pm25_pred - 1.5 * uncertainty)
            pm25_upper  = pm25_pred + 1.5 * uncertainty

            risk = classify_risk(pm25_pred)
            predictions.append({
                "target_time":  target_ts,
                "horizon_h":    h,
                "pm25_pred":    round(pm25_pred, 2),
                "pm25_lower":   round(pm25_lower, 2),
                "pm25_upper":   round(pm25_upper, 2),
                "risk_label":   risk["label"],
                "risk_level":   risk["risk"],
                "risk_color":   risk["color"],
                "action":       risk["action"],
            })

            # Extend history with synthetic row so next iteration can build lags
            new_row = df.iloc[-1].copy()
            new_row["timestamp"] = target_ts
            new_row["pm25"]      = pm25_pred
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        except Exception as exc:
            log.error(f"Forecast step h={h} failed: {exc}")
            break

    fc_df = pd.DataFrame(predictions)
    log.info(f"Forecast generated: {len(fc_df)} steps ({horizon_h}h horizon)")
    return fc_df


# ── Top-level inference runner ────────────────────────────────────────────────

def run_inference(
    horizon_h: int = 24,
    use_db: bool = False,
    days_history: int = 30,
    persist: bool = False,
) -> dict:
    """
    End-to-end inference run.

    Args:
        horizon_h:     Forecast horizon in hours.
        use_db:        Load history from DuckDB (vs. CSV/API fetch).
        days_history:  How many days of history to load for feature building.
        persist:       Write predictions to DuckDB forecasts table.

    Returns:
        {
          "current":  {pm25, risk_label, risk_level, ...},
          "forecast": pd.DataFrame,
          "recommendations": {health, policy},
          "model_name": str,
          "generated_at": str,
        }
    """
    import joblib

    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run: python -m src.train"
        )

    data = joblib.load(model_path)
    model        = data["model"]
    model_name   = data["name"]
    feature_names= data["features"]

    # Load history
    if use_db:
        from src.database import load_readings
        df_history = load_readings(days=days_history)
        if df_history.empty:
            raise RuntimeError("DB is empty — run ETL pipeline first")
    else:
        from src.data_pipeline import fetch_all
        df_history = fetch_all(days=days_history, save=False)

    # Current reading
    current_pm25 = float(df_history["pm25"].iloc[-1])
    current_risk = classify_risk(current_pm25)
    recommendations = get_recommendations(current_risk["risk"])

    # Forecast
    fc_df = forecast_horizon(model, model_name, feature_names, df_history, horizon_h)

    # Persist to DB
    if persist and not fc_df.empty:
        try:
            from src.database import save_forecast
            for _, row in fc_df.iterrows():
                save_forecast(
                    model_name=model_name,
                    target_time=row["target_time"],
                    pm25_pred=row["pm25_pred"],
                    horizon_h=int(row["horizon_h"]),
                    pm25_lower=row["pm25_lower"],
                    pm25_upper=row["pm25_upper"],
                )
        except Exception as exc:
            log.warning(f"Forecast persist failed: {exc}")

    return {
        "current":         {**current_risk},
        "forecast":        fc_df,
        "recommendations": recommendations,
        "model_name":      model_name,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    result = run_inference(horizon_h=24)
    print(f"\nCurrent: PM2.5={result['current']['pm25']} μg/m³  Risk={result['current']['risk_label']}")
    print(f"Health: {result['recommendations']['health']}")
    print(f"\n24-hour forecast:\n{result['forecast'][['target_time','pm25_pred','risk_label']].to_string(index=False)}")
