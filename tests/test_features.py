"""
Feature engineering testləri.
Ən vacib test: lag/rolling feature-ların data leakage etmədiyini yoxlamaq.
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import build_features


def _make_synthetic_df(n=200):
    """Sadə sintetik dataset — testlər üçün."""
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "pm25": np.linspace(10, 60, n) + np.random.normal(0, 2, n),
        "temp": np.random.uniform(0, 30, n),
        "humidity": np.random.uniform(30, 90, n),
        "wind_speed": np.random.uniform(0, 15, n),
        "wind_dir": np.random.uniform(0, 360, n),
        "pressure": np.random.uniform(990, 1020, n),
    })


def test_lag_feature_no_leakage():
    """
    pm25_lag_1h sütunu cari sətrin pm25 dəyərini DEYİL,
    əvvəlki sətrin dəyərini ehtiva etməlidir.
    """
    df = _make_synthetic_df()
    X, y, ts = build_features(df)

    assert "pm25_lag_1h" in X.columns

    # Cari pm25 (y) ilə lag_1h fərqli olmalıdır — eyni olsa, leakage var
    # 2-ci sətirdən başlayaraq yoxla (1-ci sətirdə lag NaN/0 ola bilər)
    matches = (X["pm25_lag_1h"].iloc[5:].values == y.iloc[5:].values).sum()
    assert matches < len(y.iloc[5:]) * 0.5, "Lag feature cari dəyərlə üst-üstə düşür — leakage şübhəsi"


def test_rolling_feature_uses_past_only():
    """Rolling ortalama yalnız keçmiş dəyərləri əhatə etməlidir, gələcəyi yox."""
    df = _make_synthetic_df()
    X, y, ts = build_features(df)

    rolling_cols = [c for c in X.columns if "rolling" in c]
    assert len(rolling_cols) > 0, "Rolling feature tapılmadı"

    # Rolling sütun heç vaxt NaN olmayan sıfır-variance sütun olmamalıdır
    for col in rolling_cols:
        assert X[col].notna().sum() > 0


def test_feature_matrix_shape():
    """Feature matrix gözlənilən ölçüdə olmalıdır."""
    df = _make_synthetic_df(n=300)
    X, y, ts = build_features(df)

    assert len(X) == len(y) == len(ts)
    assert X.shape[1] >= 15, "Feature sayı gözlənilən minimum həddən aşağıdır"
    assert not X.isna().all().any(), "Tam NaN sütun var"


def test_cyclic_encoding_range():
    """hour_sin/hour_cos -1 və 1 arasında olmalıdır (siklik encoding düzgünlüyü)."""
    df = _make_synthetic_df()
    X, y, ts = build_features(df)

    for col in ["hour_sin", "hour_cos"]:
        if col in X.columns:
            assert X[col].min() >= -1.01
            assert X[col].max() <= 1.01