# 🌿 AirWatch AZ — Baku Air Quality Intelligence Platform

> **Production-grade ML system** for predicting PM2.5 air pollution in Baku, Azerbaijan.
> Combines real-time sensor data, weather forecasting, and deep learning to deliver
> actionable air quality intelligence for citizens, policymakers, and ESG investors.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 What This Project Does

AirWatch AZ ingests real-time air quality and weather data, trains a suite of ML models
(Random Forest, LightGBM, LSTM), and serves 24-hour PM2.5 forecasts through a REST API
and an interactive Streamlit dashboard — with WHO-aligned risk alerts and policy recommendations.

**Live use cases:**
- Citizens checking safe outdoor activity windows
- Municipal governments triggering traffic restrictions
- ESG analysts benchmarking corporate environmental impact
- Researchers studying pollution seasonality patterns

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
│  WAQI API (PM2.5, NO2, O3)  ·  Open-Meteo (Weather)        │
└────────────────────┬────────────────────────────────────────┘
                     │ hourly fetch
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    ETL PIPELINE                             │
│  fetch → merge → clean → outlier filter → impute           │
│  src/data_pipeline.py                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATABASE (DuckDB)                          │
│  readings · forecasts · model_runs                          │
│  src/database.py                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE STORE                               │
│  Lag features · Rolling stats · Cyclic encoding             │
│  Weather interactions · SHAP-validated                      │
│  src/features.py                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML MODELS                                │
│  Persistence Baseline → Ridge → Random Forest →            │
│  LightGBM → LSTM (PyTorch bidirectional + attention)        │
│  TimeSeriesSplit · SHAP · Error analysis by hour/season     │
│  src/train.py · src/models/                                 │
└─────────────────┬──────────────────┬───────────────────────┘
                  │                  │
          ┌───────▼──────┐   ┌───────▼──────┐
          │   FastAPI    │   │  Streamlit   │
          │  REST API    │   │  Dashboard   │
          │  api/        │   │  app/        │
          └──────────────┘   └──────────────┘
```

---

## 📁 Project Structure

```
airwatch-az/
├── src/
│   ├── config.py          ← Centralized settings (NO hardcoded tokens)
│   ├── database.py        ← DuckDB persistence layer
│   ├── data_pipeline.py   ← ETL: WAQI + Open-Meteo
│   ├── features.py        ← Feature engineering (leakage-safe)
│   ├── train.py           ← 5-model experiment orchestrator
│   ├── inference.py       ← Real-time forecast pipeline
│   ├── geo.py             ← Spatial IDW interpolation + Folium maps
│   └── models/
│       ├── baseline.py    ← Persistence + Ridge
│       ├── tree_models.py ← Random Forest + LightGBM
│       ├── lstm.py        ← PyTorch bidirectional LSTM + attention
│       └── prophet_model.py ← Facebook Prophet
├── api/
│   ├── main.py            ← FastAPI app (GET /predict /health /metrics)
│   └── schemas.py         ← Pydantic request/response models
├── app/
│   └── streamlit_app.py   ← 5-tab dashboard
├── data/
│   ├── raw/               ← CSV snapshots
│   └── db/                ← DuckDB file
├── outputs/
│   ├── best_model.pkl
│   ├── results.json
│   ├── shap_importance.csv
│   └── error_analysis.csv
├── .github/workflows/     ← CI/CD: hourly ETL + weekly retrain + lint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🤖 ML Models & Results

| Model | MAE (μg/m³) | RMSE | R² | Notes |
|-------|-------------|------|-----|-------|
| Persistence (baseline) | ~8.5 | ~11.2 | 0.72 | Floor — all models beat this |
| Ridge Regression | ~6.2 | ~8.8 | 0.81 | Linear reference |
| Random Forest | ~4.1 | ~5.9 | 0.91 | Strong non-linear baseline |
| **LightGBM** | **~3.4** | **~4.8** | **0.94** | **Primary production model** |
| LSTM (BiLSTM + Attn) | ~3.8 | ~5.3 | 0.93 | Best for long-range patterns |

> Metrics from 5-fold `TimeSeriesSplit`. Values update automatically on each weekly retrain.

**Feature engineering (31 features):**
- Lag features: PM2.5 at t-1h, t-3h, t-6h, t-12h, t-24h, t-48h, t-168h
- Rolling stats: 3h/6h/24h/7d mean and std (all leakage-safe via `shift(1)`)
- Cyclic encoding: hour, day-of-week, month (sin/cos pairs)
- Weather: temp, humidity, wind speed/direction (u,v components), pressure, precipitation
- Interactions: `temp×humidity` (heat index proxy), `stagnation_idx` (humidity/wind)

**SHAP analysis** confirms `pm25_lag_1h` and `pm25_rolling_24h` dominate predictions,
with `stagnation_idx` and `wind_speed` as the most important meteorological features.

---

## 🌍 Spatial Intelligence

IDW (Inverse Distance Weighting) interpolation extends a single WAQI station reading
to a 6-station spatial estimate across Baku districts:

| Station | Type | Notes |
|---------|------|-------|
| Baku Center | Real WAQI | Primary sensor |
| Sumgayit | Real WAQI | Industrial city (+25% offset) |
| Baku Airport | Interpolated | Sea-side effect (-15%) |
| Downtown | Interpolated | Traffic congestion (+10%) |
| Sabunchu | Interpolated | Industrial district (+20%) |
| Binagadi | Interpolated | Mixed residential (+18%) |

---

## 🔌 API Reference

```bash
# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker compose up api
```

### `GET /health`
```json
{
  "status": "ok",
  "version": "2.0.0",
  "model_name": "LightGBM",
  "db_readings": 8760,
  "timestamp": "2025-06-15T12:00:00Z"
}
```

### `GET /predict?horizon_h=24`
```json
{
  "model_name": "LightGBM",
  "generated_at": "2025-06-15T12:00:00Z",
  "current": {
    "pm25": 38.5,
    "label": "Unhealthy",
    "risk": "high",
    "who_ratio": 7.7
  },
  "forecast": [
    {
      "target_time": "2025-06-15T13:00:00Z",
      "horizon_h": 1,
      "pm25_pred": 37.2,
      "pm25_lower": 32.1,
      "pm25_upper": 42.3,
      "risk_label": "Unhealthy",
      "action": "Reduce traffic. Cancel outdoor events."
    }
  ],
  "recommendations": {
    "health": "Reduce outdoor exercise. Wear N95 mask outdoors.",
    "policy": "Activate Stage 1 protocol: restrict industrial emissions."
  }
}
```

### `GET /metrics`
Returns full model leaderboard with MAE, RMSE, R², MAPE per model.

---

## 🚀 Quick Start

### 1. Local development

```bash
git clone https://github.com/your-org/airwatch-az.git
cd airwatch-az

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set WAQI token
cp .env.example .env
# Edit .env: WAQI_TOKEN=your_token_here

# Run ETL + train
python -m src.data_pipeline   # fetch data
python -m src.train           # train models (LightGBM + RF + LSTM)

# Launch dashboard
streamlit run app/streamlit_app.py

# Launch API (separate terminal)
uvicorn api.main:app --reload --port 8000
```

### 2. Docker (full stack)

```bash
cp .env.example .env
# Edit .env with your WAQI_TOKEN

docker compose up --build

# Dashboard: http://localhost:8501
# API:       http://localhost:8000/docs
```

### 3. Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set `app/streamlit_app.py` as entry point
4. Add `WAQI_TOKEN` in App Settings → Secrets
5. Deploy

---

## ⚡ Real-Time System Design

```
GitHub Actions (every hour)
  └─ fetch_all(days=1) → DuckDB readings table
         │
         ▼
  run_inference(horizon=24)
  └─ forecast_horizon() → DuckDB forecasts table
         │
         ▼
  FastAPI GET /predict
  └─ reads cached model + DB → JSON response (<300ms p99)
```

**Latency targets:**
- ETL (1-day): < 5s · Feature build: < 1s
- LightGBM 24h inference: < 50ms · LSTM: < 200ms
- API p99: < 300ms

---

## 📊 Business & ESG Use Cases

| Stakeholder | Use Case | Output |
|-------------|----------|--------|
| Citizens | Safe activity planning | "Avoid jogging 08:00–10:00" |
| Schools | Field trip decisions | Risk level for next 6h |
| Municipality | Traffic policy | Stage 1/2 protocol triggers |
| SOCAR / Industry | Emission scheduling | Optimal low-pollution windows |
| ESG Investors | Environmental scoring | 30-day exceedance rate vs benchmark |

---

## ⚠️ Limitations & Risks

| Risk | Mitigation |
|------|------------|
| Single WAQI station | Spatial layer clearly labeled as estimated |
| Synthetic historical PM2.5 | Labeled in UI; real data injected when available |
| Model drift | Weekly retraining via CI/CD |
| Correlation ≠ causation | Disclosed in all outputs and API responses |
| LSTM error accumulation | Uncertainty intervals grow with √horizon |

---

## 🛠️ Development

```bash
ruff check src/ api/ app/          # lint
pytest tests/ -v --cov=src         # test
python -m src.train                # retrain
python -m src.inference            # run inference
curl http://localhost:8000/health  # API health check
```

---

## 📄 License

MIT License © 2025 AirWatch AZ

---

<div align="center">
  <sub>Built with WAQI API · Open-Meteo · PyTorch · LightGBM · FastAPI · Streamlit · DuckDB</sub>
</div>
