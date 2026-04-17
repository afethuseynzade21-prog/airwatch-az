"""
AirWatch AZ — Data Pipeline
============================
WAQI API  → PM2.5, NO2, O3  (current reading)
Open-Meteo → temperature, wind, humidity (hourly archive)

Usage:
    from src.data_pipeline import fetch_all
    df = fetch_all(days=365)
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from src.config import (
    BAKU_LAT,
    BAKU_LON,
    DATA_DIR,
    WAQI_STATIONS,
    get_waqi_token,
)

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. WAQI — Air Quality Data
# ════════════════════════════════════════════════════════════════════════════

def fetch_waqi_current(station_id: str, token: str) -> dict | None:
    """Fetch the current AQI reading from a single WAQI station."""
    url = f"https://api.waqi.info/feed/{station_id}/?token={token}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            log.warning(f"WAQI status != ok for {station_id}: {data.get('data')}")
            return None

        iaqi = data["data"]["iaqi"]
        return {
            "timestamp": pd.to_datetime(data["data"]["time"]["iso"]),
            "station":   station_id,
            "pm25":      iaqi.get("pm25", {}).get("v"),
            "pm10":      iaqi.get("pm10", {}).get("v"),
            "no2":       iaqi.get("no2",  {}).get("v"),
            "o3":        iaqi.get("o3",   {}).get("v"),
            "aqi":       int(data["data"]["aqi"]) if data["data"]["aqi"] != "-" else None,
        }
    except Exception as exc:
        log.error(f"WAQI fetch error ({station_id}): {exc}")
        return None


def fetch_waqi_all_stations() -> pd.DataFrame:
    """
    Fetch current readings from all configured stations.
    Returns an empty DataFrame if no token is configured.
    """
    token = get_waqi_token()
    if not token:
        log.warning("No WAQI token configured — skipping real-time fetch")
        return pd.DataFrame()

    records = []
    for name, station_id in WAQI_STATIONS.items():
        rec = fetch_waqi_current(station_id, token)
        if rec:
            records.append(rec)
        time.sleep(0.3)   # respect rate limits

    if not records:
        log.warning("No WAQI data received from any station")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    log.info(f"WAQI: {len(df)} live readings from {len(records)} stations")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. Open-Meteo — Weather Archive (free, no key required)
# ════════════════════════════════════════════════════════════════════════════

def fetch_weather(days: int = 365) -> pd.DataFrame:
    """
    Pull hourly historical weather for Baku from Open-Meteo.
    Falls back to synthetic data if the API is unavailable.
    """
    # Open-Meteo archive has ~7-day lag
    end_date   = (datetime.now() - timedelta(days=7)).date()
    start_date = end_date - timedelta(days=days)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   BAKU_LAT,
        "longitude":  BAKU_LON,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "surface_pressure",
        ]),
        "timezone": "Asia/Baku",
    }

    try:
        log.info(f"Open-Meteo: fetching {start_date} → {end_date}")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]

        df = pd.DataFrame({
            "timestamp":  pd.to_datetime(hourly["time"]),
            "temp":       hourly["temperature_2m"],
            "humidity":   hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_dir":   hourly["wind_direction_10m"],
            "precip":     hourly["precipitation"],
            "pressure":   hourly["surface_pressure"],
        })
        log.info(f"Open-Meteo: {len(df):,} rows ({df.timestamp.min()} → {df.timestamp.max()})")
        return df

    except Exception as exc:
        log.error(f"Open-Meteo error: {exc} — using synthetic fallback")
        return _synthetic_weather(days)


def _synthetic_weather(days: int) -> pd.DataFrame:
    """Realistic synthetic weather for Baku (fallback only)."""
    idx = pd.date_range(end=datetime.now(), periods=days * 24, freq="h")
    rng = np.random.default_rng(42)
    doy = idx.dayofyear

    # Seasonal temperature: Baku averages ~14°C annual, hot summers
    temp_seasonal = 14 + 12 * np.sin(2 * np.pi * (doy - 80) / 365)
    temp = temp_seasonal + rng.normal(0, 3, len(idx))

    return pd.DataFrame({
        "timestamp":  idx,
        "temp":       temp.round(1),
        "humidity":   np.clip(60 + 15 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 8, len(idx)), 20, 95).round(1),
        "wind_speed": np.maximum(0, rng.exponential(4, len(idx))).round(1),
        "wind_dir":   rng.uniform(0, 360, len(idx)).round(0),
        "precip":     np.maximum(0, rng.normal(0.05, 0.3, len(idx))).round(2),
        "pressure":   (1013 + rng.normal(0, 5, len(idx))).round(1),
    })


# ════════════════════════════════════════════════════════════════════════════
# 3. Synthetic PM2.5 (for historical backfill when WAQI history is unavailable)
# ════════════════════════════════════════════════════════════════════════════

def _synthetic_pm25(index: pd.DatetimeIndex, weather_df: pd.DataFrame | None = None) -> pd.Series:
    """
    Physics-informed synthetic PM2.5 for Baku.
    Incorporates hour-of-day, weekday, season, and — when available —
    temperature and wind speed effects.
    """
    rng = np.random.default_rng(42)
    n   = len(index)

    # Rush-hour peak at 08:00 and 18:00
    hour_effect  = 8 * (
        np.exp(-0.5 * ((index.hour - 8) / 2) ** 2) +
        np.exp(-0.5 * ((index.hour - 18) / 2) ** 2)
    )
    week_effect  = -4 * (index.dayofweek >= 5).astype(float)
    season_effect= 12 * np.sin(2 * np.pi * (index.dayofyear - 15) / 365)  # winter peak
    noise        = rng.normal(0, 3.5, n)

    pm25 = 32 + hour_effect + week_effect + season_effect + noise

    if weather_df is not None and len(weather_df) == n:
        # Temperature inversion: low temp → pollution trapped
        temp_effect = -0.3 * weather_df["temp"].values
        # Wind dispersal: high wind → lower PM2.5
        wind_effect = -1.5 * np.minimum(weather_df["wind_speed"].values, 15)
        # Rain washout
        rain_effect = -8 * (weather_df["precip"].values > 0.5).astype(float)
        pm25 += temp_effect + wind_effect + rain_effect

    return pd.Series(np.clip(pm25, 5, 200).round(1), index=index)


# ════════════════════════════════════════════════════════════════════════════
# 4. Merge & Clean
# ════════════════════════════════════════════════════════════════════════════

def merge_and_clean(df_wx: pd.DataFrame, df_waqi: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge weather backbone with WAQI point readings.
    Strategy:
      - Open-Meteo provides the full hourly time axis (backbone).
      - WAQI current readings are joined by nearest timestamp.
      - Remaining PM2.5 gaps are filled with synthetic data.
    """
    df_wx = df_wx.copy()
    df_wx["timestamp"] = pd.to_datetime(df_wx["timestamp"]).dt.tz_localize(None).dt.floor("h")
    df = df_wx.sort_values("timestamp").reset_index(drop=True)

    # Attach real WAQI data where available
    if df_waqi is not None and not df_waqi.empty:
        df_waqi = df_waqi.copy()
        df_waqi["timestamp"] = pd.to_datetime(df_waqi["timestamp"]).dt.tz_localize(None).dt.floor("h")
        waqi_cols = [c for c in ["timestamp", "pm25", "pm10", "no2", "o3", "aqi"] if c in df_waqi.columns]
        df = pd.merge(df, df_waqi[waqi_cols], on="timestamp", how="left")
        log.info(f"WAQI join: {df['pm25'].notna().sum()} real PM2.5 points merged")
    else:
        for col in ("pm25", "pm10", "no2", "o3", "aqi"):
            df[col] = np.nan

    # Fill PM2.5 gaps with physics-informed synthetic
    missing_mask = df["pm25"].isna()
    if missing_mask.any():
        synth = _synthetic_pm25(
            pd.DatetimeIndex(df.loc[missing_mask, "timestamp"]),
            df.loc[missing_mask, ["temp", "wind_speed", "precip"]].reset_index(drop=True)
                if all(c in df.columns for c in ["temp", "wind_speed", "precip"]) else None,
        )
        df.loc[missing_mask, "pm25"] = synth.values
        log.info(f"Synthetic fill: {missing_mask.sum():,} PM2.5 rows generated")

    # Fill NO2/O3 gaps from synthetic ratios
    if df["no2"].isna().any():
        df["no2"] = df["no2"].fillna(df["pm25"] * 1.1 + np.random.default_rng(1).normal(0, 5, len(df)))
        df["no2"] = df["no2"].clip(lower=0).round(1)
    if df["o3"].isna().any():
        df["o3"] = df["o3"].fillna(df["pm25"] * 0.7 + np.random.default_rng(2).normal(0, 4, len(df)))
        df["o3"] = df["o3"].clip(lower=0).round(1)

    # ── Outlier handling (clip, not drop — preserve time axis) ───────────
    for col in ("pm25", "no2", "o3"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        upper   = q3 + 3 * iqr
        n_clip  = int((df[col] > upper).sum())
        if n_clip:
            log.info(f"  {col}: {n_clip} outliers clipped at {upper:.1f}")
        df[col] = df[col].clip(lower=0, upper=upper)

    # ── Weather imputation ────────────────────────────────────────────────
    wx_cols = ["temp", "humidity", "wind_speed", "wind_dir", "precip", "pressure"]
    df[wx_cols] = df[wx_cols].ffill().bfill()

    # ── Drop residual NaN ─────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["pm25"]).reset_index(drop=True)
    if (dropped := before - len(df)):
        log.warning(f"Dropped {dropped} rows with NaN PM2.5")

    log.info(f"Dataset ready: {len(df):,} rows | {df.timestamp.min()} → {df.timestamp.max()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 5. Main entry point
# ════════════════════════════════════════════════════════════════════════════

def fetch_all(
    days: int = 365,
    save: bool = True,
    persist_db: bool = False,
) -> pd.DataFrame:
    """
    Full ETL pipeline: APIs → merge → clean → (optional) persist.

    Args:
        days:       How many days of historical data to fetch.
        save:       Save CSV snapshot to data/raw/.
        persist_db: Also write to DuckDB (requires duckdb installed).

    Returns:
        Cleaned DataFrame with PM2.5 + weather features.
    """
    log.info("=" * 55)
    log.info("AirWatch AZ — ETL Pipeline")
    log.info("=" * 55)

    df_waqi = fetch_waqi_all_stations()
    df_wx   = fetch_weather(days)
    df      = merge_and_clean(df_wx, df_waqi if not df_waqi.empty else None)

    if save:
        out = DATA_DIR / f"baku_airquality_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(out, index=False)
        log.info(f"CSV saved: {out}")

    if persist_db:
        try:
            from src.database import upsert_readings
            is_demo = df_waqi.empty
            upsert_readings(df, is_demo=is_demo)
        except Exception as exc:
            log.warning(f"DB persist skipped: {exc}")

    log.info(f"ETL complete: {len(df):,} rows ready")
    return df


if __name__ == "__main__":
    df = fetch_all(days=365)
    print(df.tail(3).to_string())
    print(df.describe().round(2))
