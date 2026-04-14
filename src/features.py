"""
AirWatch AZ — Feature Engineering
===================================
Bütün feature-lar leakage-safe-dir:
  - shift(1) ilə sürüşdürülüb
  - t anında yalnız t-1 məlumat istifadə olunur

İstifadə:
    from src.features import build_features
    X, y = build_features(df)
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

TARGET = "pm25"


def build_features(df: pd.DataFrame, target_col: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """
    Xam datasetdən model feature-ları hazırla.

    Args:
        df: merge_and_clean() çıxışı
        target_col: hədəf sütun

    Returns:
        X: feature matrix
        y: hədəf sütun (pm25)
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    log.info(f"Feature engineering başlayır: {len(df):,} sətir")

    # ── 1. Vaxt feature-ları ─────────────────────────────────────────────
    df["hour"]      = df["timestamp"].dt.hour
    df["dow"]       = df["timestamp"].dt.dayofweek        # 0=Monday
    df["month"]     = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["is_weekend"]= (df["dow"] >= 5).astype(int)

    # Siklik encoding — 23:00 ilə 01:00 "yaxın" olmalıdır
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)

    # ── 2. Lag feature-ları (leakage-safe: shift ilə) ────────────────────
    # ⚠️ shift(1) vacibdir — onsuz gələcək məlumat training-ə keçir
    df["pm25_lag_1h"]   = df["pm25"].shift(1)
    df["pm25_lag_3h"]   = df["pm25"].shift(3)
    df["pm25_lag_6h"]   = df["pm25"].shift(6)
    df["pm25_lag_12h"]  = df["pm25"].shift(12)
    df["pm25_lag_24h"]  = df["pm25"].shift(24)
    df["pm25_lag_48h"]  = df["pm25"].shift(48)
    df["pm25_lag_168h"] = df["pm25"].shift(168)    # 1 həftə əvvəl

    # ── 3. Rolling statistika (leakage-safe: shift(1) əlavə edilib) ──────
    df["pm25_rolling_3h"]  = df["pm25"].shift(1).rolling(3).mean()
    df["pm25_rolling_6h"]  = df["pm25"].shift(1).rolling(6).mean()
    df["pm25_rolling_24h"] = df["pm25"].shift(1).rolling(24).mean()
    df["pm25_rolling_7d"]  = df["pm25"].shift(1).rolling(168).mean()    # 7 gün

    # Rolling std — dəyişkənliyi ölçür
    df["pm25_std_24h"]     = df["pm25"].shift(1).rolling(24).std()
    df["pm25_std_7d"]      = df["pm25"].shift(1).rolling(168).std()

    # ── 4. Hava feature-ları ──────────────────────────────────────────────
    # Külək — dispersiyaya təsir edir
    df["wind_speed"]  = df["wind_speed"]
    df["wind_u"]      = df["wind_speed"] * np.sin(np.radians(df["wind_dir"]))
    df["wind_v"]      = df["wind_speed"] * np.cos(np.radians(df["wind_dir"]))

    # Temperatur inversiyası — aşağı temp → çirklənmə yığılır
    df["temp"]        = df["temp"]
    df["temp_sq"]     = df["temp"] ** 2               # nonlinear effekt üçün

    # Nəmlik
    df["humidity"]    = df["humidity"]

    # ── 5. Interaction feature-ları ──────────────────────────────────────
    # temp × humidity — birlikdə effekti güclüdür
    df["temp_x_humidity"]   = df["temp"] * df["humidity"] / 100

    # Yüksək nəmlik + aşağı külək → çirklənmə tələsi
    df["stagnation_idx"]    = df["humidity"] / (df["wind_speed"].clip(lower=0.1))

    # ── 6. Əlavə kontekst feature-ları ───────────────────────────────────
    if "no2" in df.columns:
        df["no2_lag_1h"]  = df["no2"].shift(1)
        df["no2_lag_24h"] = df["no2"].shift(24)

    if "precip" in df.columns:
        df["precip_lag_3h"] = df["precip"].shift(1).rolling(3).sum()   # yağış PM-i azaldır

    if "pressure" in df.columns:
        df["pressure"] = df["pressure"]
        df["pressure_change_3h"] = df["pressure"].shift(1).diff(3)     # hava dəyişimi

    # ── 7. Yekun feature siyahısı ─────────────────────────────────────────
    feature_cols = [
        # vaxt
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "is_weekend",
        # lag
        "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_12h",
        "pm25_lag_24h", "pm25_lag_48h", "pm25_lag_168h",
        # rolling
        "pm25_rolling_3h", "pm25_rolling_6h", "pm25_rolling_24h", "pm25_rolling_7d",
        "pm25_std_24h", "pm25_std_7d",
        # hava
        "temp", "temp_sq", "humidity", "wind_speed", "wind_u", "wind_v",
        # interaction
        "temp_x_humidity", "stagnation_idx",
    ]

    # Mövcud olan opsional sütunları əlavə et
    optional = ["no2_lag_1h", "no2_lag_24h", "precip_lag_3h", "pressure", "pressure_change_3h"]
    feature_cols += [c for c in optional if c in df.columns]

    # NaN sətirləri sil (lag-lar səbəbindən ilk 168 saat boş olacaq)
    df_model = df[feature_cols + [target_col, "timestamp"]].dropna()
    pct_lost  = (1 - len(df_model) / len(df)) * 100 if len(df) > 0 else 0
    log.info(f"NaN silindikdən sonra: {len(df_model):,} sətir ({pct_lost:.1f}% itirildi)")

    X = df_model[feature_cols]
    y = df_model[target_col]

    log.info(f"Feature matrix: {X.shape[0]:,} sətir × {X.shape[1]} feature")
    return X, y, df_model["timestamp"]


def feature_importance_summary(model, feature_names: list) -> pd.DataFrame:
    """Random Forest / LightGBM üçün feature importance cədvəli."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        raise ValueError("Model feature_importances_ attribute-unu dəstəkləmir")

    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
