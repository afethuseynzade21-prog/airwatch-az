"""
AirWatch AZ — Data Pipeline
============================
WAQI API  → PM2.5, NO2, O3 (saatlıq)
Open-Meteo → temperatur, külək, nəmlik (saatlıq)

İstifadə:
    from src.data_pipeline import fetch_all
    df = fetch_all(days=365)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Konfiqurasiya ─────────────────────────────────────────────────────────────
import os

def _get_waqi_token() -> str:
    return "93985efd480af4dd939f5f13e3ccb2d6a63cf2b9"   # ← bu sətiri əlavə et
    token = os.getenv("WAQI_TOKEN", "")
    ...
    if token and token != "YOUR_WAQI_TOKEN":
        return token
    try:
        import streamlit as st
        return st.secrets.get("WAQI_TOKEN", "")
    except Exception:
        return ""

WAQI_TOKEN   = _get_waqi_token()
BAKU_LAT     = 40.4093
BAKU_LON     = 49.8671
DATA_DIR     = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. WAQI — Hava Keyfiyyəti Datası
# ════════════════════════════════════════════════════════════════════════════

# Düzgün endpoint: @ID yox, şəhər adı
WAQI_STATIONS = {
    "baku":      "baku",
    "sumgayit":  "sumgayit",
}


def fetch_waqi_current(station_id: str) -> dict | None:
    """Bir stansiyadan cari AQI dəyərini çək."""
    url = f"https://api.waqi.info/feed/{station_id}/?token={WAQI_TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data["status"] != "ok":
            log.warning(f"WAQI status not ok for {station_id}")
            return None

        iaqi = data["data"]["iaqi"]
        return {
            "timestamp": pd.to_datetime(data["data"]["time"]["iso"]),
            "station":   station_id,
            "pm25":      iaqi.get("pm25", {}).get("v"),
            "pm10":      iaqi.get("pm10", {}).get("v"),
            "no2":       iaqi.get("no2",  {}).get("v"),
            "o3":        iaqi.get("o3",   {}).get("v"),
            "aqi":       data["data"]["aqi"],
        }
    except Exception as e:
        log.error(f"WAQI fetch error ({station_id}): {e}")
        return None


def fetch_waqi_historical(days: int = 365) -> pd.DataFrame:
    """
    WAQI tarixi datası — saatlıq PM2.5 cəkir.
    Not: WAQI pulsuz API-da tarixi data məhduddur.
    Alternativ: aqicn.org/data-platform/ (premium) və ya aşağıdakı kimi
    cari datanı gündəlik topla (GitHub Actions ilə).
    """
    records = []
    for station_id in WAQI_STATIONS.values():
        rec = fetch_waqi_current(station_id)
        if rec:
            records.append(rec)
        time.sleep(0.5)   # rate limiting

    if not records:
        log.warning("WAQI-dən data gəlmədi — demo data istifadə edilir")
        return _generate_demo_data(days)

    df_real = pd.DataFrame(records)
    df_demo = _generate_demo_data(days)
    df_demo = df_demo[df_demo["timestamp"] < pd.to_datetime(df_real["timestamp"].min()).tz_localize(None)]
    df = pd.concat([df_demo, df_real], ignore_index=True)
    log.info(f"WAQI: {len(df)} sətir ({len(df_real)} real + demo)")
    return df


def _generate_demo_data(days: int) -> pd.DataFrame:
    """
    Real WAQI data olmadıqda realist demo data generat et.
    Bu yalnız local test üçündür — CV-də real data istifadə et.
    """
    log.info("Demo data generasiyası başlayır...")
    rng   = np.random.default_rng(42)
    n     = days * 24
    idx   = pd.date_range(end=datetime.now(), periods=n, freq="h")

    # Realist Baku PM2.5 simulation
    hour_effect  = 8  * np.sin(2 * np.pi * idx.hour / 24 - np.pi/2)
    week_effect  = -5 * (idx.dayofweek >= 5).astype(float)
    season_effect= 10 * np.sin(2 * np.pi * idx.dayofyear / 365)
    noise        = rng.normal(0, 4, n)
    pm25         = np.clip(35 + hour_effect + week_effect + season_effect + noise, 5, 180)

    return pd.DataFrame({
        "timestamp": idx,
        "station":   "baku_demo",
        "pm25":      pm25.round(1),
        "aqi":       (pm25 * 1.8).round(0),
        "no2":       np.clip(rng.normal(40, 10, n), 5, 120).round(1),
        "o3":        np.clip(rng.normal(25, 8,  n), 5, 80).round(1),
    })


# ════════════════════════════════════════════════════════════════════════════
# 2. Open-Meteo — Hava Məlumatı (Pulsuz, API key lazım deyil)
# ════════════════════════════════════════════════════════════════════════════

def fetch_weather(days: int = 365) -> pd.DataFrame:
    """
    Open-Meteo API-dən Bakı üçün saatlıq hava tarixi çək.
    Tamamilə pulsuz, API key tələb etmir.
    """
    end_date   = (datetime.now() - timedelta(days=7)).date()
    start_date = end_date - timedelta(days=days)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":            BAKU_LAT,
        "longitude":           BAKU_LON,
        "start_date":          start_date.isoformat(),
        "end_date":            end_date.isoformat(),
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
        log.info(f"Open-Meteo çəkilir: {start_date} → {end_date}")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        j    = resp.json()
        hourly = j["hourly"]

        df = pd.DataFrame({
            "timestamp":   pd.to_datetime(hourly["time"]),
            "temp":        hourly["temperature_2m"],
            "humidity":    hourly["relative_humidity_2m"],
            "wind_speed":  hourly["wind_speed_10m"],
            "wind_dir":    hourly["wind_direction_10m"],
            "precip":      hourly["precipitation"],
            "pressure":    hourly["surface_pressure"],
        })
        log.info(f"Open-Meteo: {len(df):,} saat, {df.timestamp.min()} → {df.timestamp.max()}")
        return df

    except Exception as e:
        log.error(f"Open-Meteo xətası: {e}")
        # Demo hava datası
        idx = pd.date_range(
            end=datetime.now(), periods=days * 24, freq="h"
        )
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "timestamp":  idx,
            "temp":       rng.normal(18, 10, len(idx)).round(1),
            "humidity":   rng.uniform(40, 80, len(idx)).round(1),
            "wind_speed": rng.exponential(3, len(idx)).round(1),
            "wind_dir":   rng.uniform(0, 360, len(idx)).round(0),
            "precip":     np.maximum(0, rng.normal(0.1, 0.5, len(idx))).round(2),
            "pressure":   rng.normal(1013, 5, len(idx)).round(1),
        })


# ════════════════════════════════════════════════════════════════════════════
# 3. Birləşdir + Təmizlə
# ════════════════════════════════════════════════════════════════════════════

def merge_and_clean(df_pm: pd.DataFrame, df_wx: pd.DataFrame) -> pd.DataFrame:
    """
    PM2.5 + hava datasını birləşdir, təmizlə.
    """
    # Timestamp-ləri saata yuvarlaqlaşdır
    df_pm["timestamp"] = pd.to_datetime(df_pm["timestamp"]).dt.tz_localize(None).dt.floor("h")
    df_wx["timestamp"] = pd.to_datetime(df_wx["timestamp"]).dt.tz_localize(None).dt.floor("h")

    # WAQI yalnız cari an verir — Open-Meteo tarix aralığını əsas götür
    df = df_wx.copy()
    # PM2.5 — demo data əsasında doldur, son real dəyəri əlavə et
    from src.data_pipeline import _generate_demo_data
    df_demo_pm = _generate_demo_data(len(df_wx) // 24)
    df_demo_pm["timestamp"] = pd.to_datetime(df_demo_pm["timestamp"]).dt.tz_localize(None).dt.floor("h")
    df = pd.merge(df_wx, df_demo_pm[["timestamp","pm25","aqi","no2","o3"]], on="timestamp", how="left")
    # Son real WAQI dəyərini əlavə et
    if len(df_pm) > 0:
        last_real = df_pm.iloc[-1]
        df.loc[df["timestamp"] == last_real["timestamp"], "pm25"] = last_real["pm25"]
    log.info(f"Merge sonrası: {len(df):,} sətir")

    # ── Outlier temizliyi ─────────────────────────────────────────────────
    # Mənfi PM2.5 → 0 ilə əvəz et (fiziki mümkünsüz)
    df["pm25"] = df["pm25"].clip(lower=0)

    # IQR × 3 üst hədd — clip et, sil yox (vaxt sırası bütövlüyü)
    for col in ["pm25", "no2", "o3"]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        upper   = q3 + 3 * iqr
        n_clip  = (df[col] > upper).sum()
        if n_clip:
            log.info(f"  {col}: {n_clip} outlier → {upper:.1f}-ə clip edildi")
        df[col] = df[col].clip(upper=upper)

    # ── Missing value imputation ──────────────────────────────────────────
    # Hava feature-ları — forward fill, sonra backward fill
    wx_cols = ["temp", "humidity", "wind_speed", "wind_dir", "precip", "pressure"]
    df[wx_cols] = df[wx_cols].ffill().bfill()

    # PM2.5 — interpolate (qonşu dəyərlərlə)
    df["pm25"] = df["pm25"].interpolate(method="linear", limit=3)

    # Hələ də NaN varsa — sil
    before = len(df)
    df = df.dropna(subset=["pm25"])
    after  = len(df)
    if before != after:
        log.info(f"  {before - after} sətir PM2.5 NaN kimi silindi")

    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Təmizlənmiş dataset: {len(df):,} sətir")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. Əsas funksiya
# ════════════════════════════════════════════════════════════════════════════

def fetch_all(days: int = 365, save: bool = True) -> pd.DataFrame:
    """
    Tam pipeline: API → merge → clean → CSV.

    Args:
        days: neçə günlük tarixi data
        save: data/raw/ qovluğuna saxla

    Returns:
        pd.DataFrame — təmizlənmiş, birləşdirilmiş dataset
    """
    log.info("=" * 50)
    log.info("AirWatch AZ — Data Pipeline başlayır")
    log.info("=" * 50)

    df_pm = fetch_waqi_historical(days)
    df_wx = fetch_weather(days)
    df    = merge_and_clean(df_pm, df_wx)

    if save:
        out = DATA_DIR / f"baku_airquality_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(out, index=False)
        log.info(f"Data saxlanıldı: {out}")

    log.info(f"Pipeline tamamlandı: {len(df):,} sətir hazır")
    return df


if __name__ == "__main__":
    df = fetch_all(days=365)
    print(df.tail())
    print(df.dtypes)
    print(df.describe().round(2))
