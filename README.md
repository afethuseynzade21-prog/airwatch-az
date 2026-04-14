# 🌿 AirWatch AZ — Baku Air Quality Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-green?logo=streamlit)](https://airwatch-az.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-gray)](LICENSE)

> **PM2.5 forecasting system for Baku, Azerbaijan** — combining ground station data,
> weather features, and machine learning to deliver actionable air quality intelligence
> for government agencies, municipalities, and ESG investors.

---

## 🎬 Demo

<!-- 60 saniyəlik Loom GIF burada -->
*[Demo GIF / Loom link əlavə et]*

**Live app:** [YOUR-APP.streamlit.app](https://YOUR-APP.streamlit.app)

---

## 📊 Model Results

> Results from `TimeSeriesSplit(n_splits=5)` on real WAQI Baku data.
> All lag/rolling features use `.shift(1)` — no data leakage.

| Model | MAE (μg/m³) | RMSE | MAPE | R² |
|-------|------------|------|------|----|
| Persistence (baseline) | X.X ± Y.Y | X.X | X.X% | X.XX |
| Ridge Regression       | X.X ± Y.Y | X.X | X.X% | X.XX |
| Random Forest          | X.X ± Y.Y | X.X | X.X% | X.XX |
| LightGBM (Phase 2)     | X.X ± Y.Y | X.X | X.X% | X.XX |

*Replace X.X values with your real results after running `python -m src.train`*

**Actual vs Predicted:**
![actual_vs_predicted](outputs/actual_vs_pred.png)

---

## 🏗️ Architecture

```
WAQI API (PM2.5, NO2)  ──┐
                          ├──► merge_and_clean() ──► build_features() ──► Model ──► Streamlit
Open-Meteo (weather)   ──┘
```

**Data flow:** Raw API → Cleaned hourly series → Lag/rolling features → TimeSeriesSplit → Best model → Dashboard

---

## 🚀 Quick Start

```bash
# 1. Clone + install
git clone https://github.com/YOUR-USER/airwatch-az
cd airwatch-az
pip install -r requirements.txt

# 2. WAQI API token al: waqi.info/api-access/
export WAQI_TOKEN="your_token_here"

# 3. Data çək
python -c "from src.data_pipeline import fetch_all; fetch_all(days=365)"

# 4. Model train et
python -m src.train

# 5. Dashboard işlət
streamlit run app/streamlit_app.py
```

---

## 📁 Repo Structure

```
airwatch-az/
├── data/raw/              ← WAQI + Open-Meteo raw CSV
├── notebooks/
│   ├── 01_eda.ipynb       ← Exploratory analysis
│   ├── 02_features.ipynb  ← Feature engineering + leakage check
│   └── 03_models.ipynb    ← TimeSeriesSplit model comparison
├── src/
│   ├── data_pipeline.py   ← WAQI + Open-Meteo API
│   ├── features.py        ← Lag, rolling, cyclic encoding
│   └── train.py           ← TimeSeriesSplit experiment runner
├── app/
│   └── streamlit_app.py   ← 3-tab dashboard
├── outputs/
│   ├── actual_vs_pred.png ← Model validation chart
│   ├── residuals.png      ← Error distribution
│   └── best_model.pkl     ← Serialised model
└── .github/workflows/
    └── daily_update.yml   ← Automated daily data refresh
```

---

## 🔬 Feature Engineering

All features are **leakage-safe** (shifted to avoid look-ahead bias):

| Feature | Formula | Why |
|---------|---------|-----|
| `pm25_lag_1h` | `pm25.shift(1)` | Strongest predictor (~0.85 correlation) |
| `pm25_rolling_7d` | `pm25.shift(1).rolling(168).mean()` | Weekly trend |
| `hour_sin/cos` | `sin(2π×hour/24)` | Cyclic time encoding |
| `temp_x_humidity` | `temp * humidity / 100` | Interaction feature |
| `stagnation_idx` | `humidity / wind_speed` | Pollution trap indicator |

---

## ⚠️ Model Limitations

1. **Baku WAQI stations:** Only 3–5 monitoring stations — spatial coverage is limited
2. **Correlation, not causation:** This model identifies statistical patterns, not causal mechanisms
3. **Validation period:** Results apply to the validation timeframe; distribution shifts may degrade performance
4. **Missing satellite data:** Phase 1 uses ground stations only. Sentinel-5P added in Phase 2.

---

## ⚠️ Causal Warning

> This model identifies **statistical correlations** between weather conditions,
> time patterns, and PM2.5 levels. It does **NOT** establish causal relationships.
> Policy decisions — emission restrictions, industrial shutdowns, traffic controls —
> **require causal validation** through controlled studies or domain expert review
> before implementation.

---

## 🗺️ Roadmap

- [x] Phase 1: WAQI + Open-Meteo pipeline, RF model, Streamlit dashboard
- [ ] Phase 2: Sentinel-5P satellite data, LightGBM + SHAP, Folium risk map
- [ ] Phase 3: Prophet 3-day forecast, GitHub Actions automation, PDF reports

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Data sources: [WAQI](https://waqi.info) · [Open-Meteo](https://open-meteo.com) · WHO PM2.5 Guidelines (2021)*

---

## 🚀 Deploy

### Streamlit Cloud (Tövsiyə edilən)

1. Bu repo-nu GitHub-a yüklə
2. [share.streamlit.io](https://share.streamlit.io) → New app
3. Repo seç → Main file: `app/streamlit_app.py`
4. **App Settings → Secrets** bölməsinə bu məzmunu əlavə et:
   ```toml
   WAQI_TOKEN = "sənin_tokenin"
   ```
5. Deploy → URL alırsın!

### WAQI Token Alma
1. [aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/)
2. E-mail daxil et → Token gəlir (anında, 24 saat gözləmək lazım deyil)
3. Test: `https://api.waqi.info/feed/baku/?token=SENİN_TOKEN`
4. `{"status":"ok","data":{...}}` görürsənsə — işləyir

### Lokal İşə Salma
```bash
git clone https://github.com/USERNAME/airwatch-az
cd airwatch-az
pip install -r requirements.txt

# .streamlit/secrets.toml yarat:
mkdir -p .streamlit
echo 'WAQI_TOKEN = "sənin_tokenin"' > .streamlit/secrets.toml

streamlit run app/streamlit_app.py
```

---
