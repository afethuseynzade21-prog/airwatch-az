"""
Risk klassifikasiyası testləri.
WHO threshold-larının düzgün tətbiq edildiyini yoxlayır.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import classify_risk


def test_good_air_quality():
    """PM2.5 ≤ 12 → 'Yaxşı' kateqoriyası."""
    result = classify_risk(8.0)
    assert result["label"] in ("Yaxşı", "Good")
    assert result["risk"] == "low"


def test_moderate_air_quality():
    """PM2.5 12-35 arası → 'Orta' kateqoriyası."""
    result = classify_risk(25.0)
    assert result["label"] in ("Orta", "Moderate")
    assert result["risk"] == "medium"


def test_unhealthy_air_quality():
    """PM2.5 35-55 arası → 'Zərərli' kateqoriyası."""
    result = classify_risk(45.0)
    assert result["label"] in ("Zərərli", "Unhealthy")
    assert result["risk"] == "high"


def test_hazardous_air_quality():
    """PM2.5 > 150 → ən yüksək risk kateqoriyası."""
    result = classify_risk(200.0)
    assert result["risk"] in ("extreme", "critical")


def test_who_ratio_calculation():
    """WHO illik norması (5 μg/m³) əsasında nisbət düzgün hesablanmalıdır."""
    result = classify_risk(25.0)
    assert result["who_ratio"] == 5.0  # 25 / 5 = 5.0


def test_boundary_values():
    """Sərhəd dəyərləri (threshold-ların düz üzərində) xəta verməməlidir."""
    for val in [0, 12, 35, 55, 150, 500]:
        result = classify_risk(float(val))
        assert "label" in result
        assert "color" in result


def test_risk_monotonicity():
    """PM2.5 artdıqca risk kateqoriyası heç vaxt azalmamalıdır."""
    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3, "extreme": 4}
    prev_level = -1
    for pm25 in [5, 20, 45, 100, 300]:
        result = classify_risk(float(pm25))
        level = risk_order.get(result["risk"], -1)
        assert level >= prev_level, f"PM2.5={pm25} üçün risk geriyə getdi"
        prev_level = level