"""
AirWatch AZ — Centralized Configuration
========================================
Single source of truth for all settings.
Never hardcode tokens or paths elsewhere.
"""

import os
import logging
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data" / "raw"
DB_DIR     = ROOT_DIR / "data" / "db"
MODEL_DIR  = ROOT_DIR / "outputs"
LOG_DIR    = ROOT_DIR / "logs"

for _d in (DATA_DIR, DB_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Baku geo ──────────────────────────────────────────────────────────────────
BAKU_LAT = 40.4093
BAKU_LON  = 49.8671

# ── WAQI Stations ─────────────────────────────────────────────────────────────
WAQI_STATIONS = {
    "baku":     "baku",
    "sumgayit": "sumgayit",
}

# Simulated station coordinates for spatial layer
STATION_COORDS = {
    "baku":               (40.4093, 49.8671),
    "sumgayit":           (40.5897, 49.6686),
    "baku_airport":       (40.4675, 50.0467),
    "baku_downtown":      (40.3777, 49.8920),
    "baku_sabunchu":      (40.4400, 49.9400),
    "baku_binagadi":      (40.4600, 49.8300),
}

# ── API tokens ────────────────────────────────────────────────────────────────

def get_waqi_token() -> str:
    """
    Resolve WAQI token from environment or Streamlit secrets.
    Priority: env var → Streamlit secrets → empty string.
    Never hardcode tokens in source files.
    """
    token = os.getenv("WAQI_TOKEN", "")
    if token and token not in ("", "YOUR_WAQI_TOKEN"):
        return token
    try:
        import streamlit as st
        token = st.secrets.get("WAQI_TOKEN", "")
        if token:
            return token
    except Exception:
        pass
    return ""

# ── Model hyperparameters ─────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":    300,
    "max_depth":       12,
    "min_samples_leaf": 5,
    "max_features":    "sqrt",
    "n_jobs":          -1,
    "random_state":    42,
}

LGB_PARAMS = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         8,
    "num_leaves":        31,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "random_state":      42,
    "verbose":           -1,
}

LSTM_PARAMS = {
    "seq_len":      24,     # 24-hour lookback window
    "hidden_size":  128,
    "num_layers":   2,
    "dropout":      0.2,
    "lr":           1e-3,
    "epochs":       50,
    "batch_size":   64,
    "patience":     8,      # early stopping
}

# ── WHO PM2.5 thresholds (μg/m³) ─────────────────────────────────────────────
WHO_THRESHOLDS = [
    {"label": "Good",         "min": 0,   "max": 12,  "color": "#2ecc71",
     "risk": "low",    "action": "No restrictions needed."},
    {"label": "Moderate",     "min": 12,  "max": 35,  "color": "#f1c40f",
     "risk": "medium", "action": "Sensitive groups should limit prolonged outdoor exertion."},
    {"label": "Unhealthy",    "min": 35,  "max": 55,  "color": "#e67e22",
     "risk": "high",   "action": "Reduce traffic. Cancel outdoor events."},
    {"label": "Very Unhealthy","min": 55, "max": 150, "color": "#e74c3c",
     "risk": "critical","action": "Restrict industrial activity. Avoid all outdoor activity."},
    {"label": "Hazardous",    "min": 150, "max": 9999,"color": "#8e44ad",
     "risk": "extreme", "action": "Emergency protocol. Evacuation of sensitive populations."},
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
