"""
AirWatch AZ — FastAPI Prediction Service
==========================================
Production REST API for PM2.5 predictions.

Endpoints:
  GET  /health                    — liveness probe
  GET  /predict?horizon_h=24      — multi-step forecast
  GET  /metrics                   — model performance metrics
  GET  /current                   — latest readings from all stations
  GET  /docs                      — Swagger UI (auto-generated)

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker compose up api
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ForecastStep,
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    CurrentResponse,
    StationReading,
)
from src.config import MODEL_DIR, WHO_THRESHOLDS
from src.inference import classify_risk, run_inference

log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AirWatch AZ API",
    description=(
        "PM2.5 air quality prediction API for Baku, Azerbaijan.\n\n"
        "Data sources: WAQI API + Open-Meteo. "
        "Models: LightGBM / LSTM / Random Forest."
    ),
    version="2.0.0",
    contact={
        "name":  "AirWatch AZ",
        "url":   "https://github.com/your-org/airwatch-az",
        "email": "contact@airwatch-az.io",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Startup — warm up model ───────────────────────────────────────────────────

_model_cache: dict = {}


def _load_model_meta() -> dict:
    """Load model metadata from disk (fast — no inference)."""
    global _model_cache
    if _model_cache:
        return _model_cache

    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        return {}

    import joblib
    data = joblib.load(model_path)
    _model_cache = {
        "model":      data["model"],
        "name":       data.get("name", "Unknown"),
        "features":   data.get("features", []),
        "n_samples":  data.get("n_samples"),
        "n_features": data.get("n_features"),
    }
    return _model_cache


@app.on_event("startup")
async def startup_event():
    meta = _load_model_meta()
    if meta:
        log.info(f"API ready — model: {meta['name']}  features: {len(meta['features'])}")
    else:
        log.warning("No trained model found. Train one with: python -m src.train")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness probe. Returns model status and DB row count."""
    meta = _load_model_meta()
    db_count = None
    try:
        from src.database import db_stats
        db_count = db_stats().get("readings")
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="2.0.0",
        model_name=meta.get("name"),
        db_readings=db_count,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(
    horizon_h: int = Query(default=24, ge=1, le=72, description="Forecast horizon (hours)"),
):
    """
    Generate a multi-step PM2.5 forecast.

    - **horizon_h**: number of hours to forecast (1–72, default 24)

    Returns current reading, full forecast table, and health/policy recommendations.
    """
    if not _load_model_meta():
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run: python -m src.train",
        )

    try:
        result = run_inference(horizon_h=horizon_h, persist=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    fc_steps = [
        ForecastStep(
            target_time=row["target_time"],
            horizon_h=int(row["horizon_h"]),
            pm25_pred=row["pm25_pred"],
            pm25_lower=row["pm25_lower"],
            pm25_upper=row["pm25_upper"],
            risk_label=row["risk_label"],
            risk_level=row["risk_level"],
            risk_color=row["risk_color"],
            action=row["action"],
        )
        for _, row in result["forecast"].iterrows()
    ]

    return PredictResponse(
        model_name=result["model_name"],
        generated_at=datetime.fromisoformat(result["generated_at"]),
        current=result["current"],
        forecast=fc_steps,
        recommendations=result["recommendations"],
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Model"])
def metrics():
    """
    Return model training metrics from the last experiment run.
    Metrics are computed with TimeSeriesSplit — no data leakage.
    """
    results_path = MODEL_DIR / "results.json"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No results.json found. Run: python -m src.train",
        )

    leaderboard = json.loads(results_path.read_text())

    meta = _load_model_meta()
    best_name = meta.get("name", "")
    best_row  = next((r for r in leaderboard if r.get("model") == best_name), {})

    return MetricsResponse(
        model_name=best_name,
        n_samples=meta.get("n_samples"),
        n_features=meta.get("n_features"),
        mae=best_row.get("mae"),
        rmse=best_row.get("rmse"),
        mape=best_row.get("mape"),
        r2=best_row.get("r2"),
        mae_std=best_row.get("mae_std"),
        rmse_std=best_row.get("rmse_std"),
        leaderboard=leaderboard,
    )


@app.get("/current", response_model=CurrentResponse, tags=["Data"])
def current():
    """
    Return the latest PM2.5 reading for all monitored stations.
    Falls back to most recent DB record if live API is unavailable.
    """
    from src.data_pipeline import fetch_waqi_all_stations
    from src.inference import classify_risk

    now = datetime.now(timezone.utc)
    stations_out = []

    df_live = fetch_waqi_all_stations()
    if df_live.empty:
        # Fallback: load from DB
        try:
            from src.database import load_latest_reading
            rec = load_latest_reading()
            if rec:
                risk = classify_risk(float(rec.get("pm25") or 0))
                stations_out.append(StationReading(
                    station=rec.get("station", "baku"),
                    timestamp=rec.get("timestamp", now),
                    pm25=rec.get("pm25"),
                    aqi=rec.get("aqi"),
                    risk_label=risk["label"],
                    risk_level=risk["risk"],
                ))
        except Exception:
            pass
    else:
        for _, row in df_live.iterrows():
            pm25 = float(row.get("pm25") or 0)
            risk = classify_risk(pm25)
            stations_out.append(StationReading(
                station=row.get("station", "unknown"),
                timestamp=row.get("timestamp", now),
                pm25=pm25 or None,
                aqi=int(row["aqi"]) if row.get("aqi") else None,
                risk_label=risk["label"],
                risk_level=risk["risk"],
            ))

    if not stations_out:
        raise HTTPException(status_code=503, detail="No station data available")

    return CurrentResponse(stations=stations_out, generated_at=now)


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
