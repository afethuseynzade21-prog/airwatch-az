"""
Data pipeline testləri.
Ən vacib: timezone merge bug-ının regress testi (2026-06-19 düzəldildi).
"""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import merge_and_clean


def test_merge_handles_mixed_timezones():
    """
    REGRESSION TEST: df_pm timezone-aware, df_wx timezone-naive olduqda
    merge_and_clean xəta verməməlidir.

    Bu test 2026-06-19-da GitHub Actions-da tapılan real bug-ı əhatə edir:
    'Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'
    """
    # Timezone-aware timestamp (real WAQI formatına bənzər)
    ts_aware = pd.date_range("2026-06-01", periods=10, freq="h", tz="Asia/Baku")
    df_pm = pd.DataFrame({
        "timestamp": ts_aware,
        "pm25": [30.0] * 10,
        "aqi": [60.0] * 10,
        "no2": [10.0] * 10,
        "o3": [20.0] * 10,
    })

    # Timezone-naive timestamp (Open-Meteo formatına bənzər)
    ts_naive = pd.date_range("2026-06-01", periods=240, freq="h")
    df_wx = pd.DataFrame({
        "timestamp": ts_naive,
        "temp": [20.0] * 240,
        "humidity": [60.0] * 240,
        "wind_speed": [5.0] * 240,
        "wind_dir": [180.0] * 240,
        "precip": [0.0] * 240,
        "pressure": [1010.0] * 240,
    })

    # Xəta verməməlidir (əvvəllər ValueError verirdi)
    result = merge_and_clean(df_pm, df_wx)
    assert result is not None
    assert "timestamp" in result.columns


def test_merge_output_no_duplicate_timestamps():
    """Merge sonrası timestamp sütununda dublikat olmamalıdır."""
    ts = pd.date_range("2026-06-01", periods=50, freq="h")
    df_pm = pd.DataFrame({"timestamp": ts[:5], "pm25": [30.0]*5, "aqi": [60.0]*5, "no2": [10.0]*5, "o3": [20.0]*5})
    df_wx = pd.DataFrame({
        "timestamp": ts, "temp": [20.0]*50, "humidity": [60.0]*50,
        "wind_speed": [5.0]*50, "wind_dir": [180.0]*50, "precip": [0.0]*50, "pressure": [1010.0]*50,
    })

    result = merge_and_clean(df_pm, df_wx)
    assert result["timestamp"].duplicated().sum() == 0


def test_merge_with_sufficient_data_window():
    """
    Kifayət qədər data pəncərəsi (≥7 gün) ilə merge sonrası
    boş olmayan dataset qaytarmalıdır.
    """
    ts = pd.date_range("2026-06-01", periods=24*10, freq="h")
    df_pm = pd.DataFrame({"timestamp": [ts[0]], "pm25": [30.0], "aqi": [60.0], "no2": [10.0], "o3": [20.0]})
    df_wx = pd.DataFrame({
        "timestamp": ts, "temp": [20.0]*len(ts), "humidity": [60.0]*len(ts),
        "wind_speed": [5.0]*len(ts), "wind_dir": [180.0]*len(ts),
        "precip": [0.0]*len(ts), "pressure": [1010.0]*len(ts),
    })

    result = merge_and_clean(df_pm, df_wx)
    assert len(result) > 0, "10 günlük pəncərə ilə belə boş dataset qaytarıldı"