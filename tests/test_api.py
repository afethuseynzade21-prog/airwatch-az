"""
FastAPI endpoint testləri.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

try:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    HAS_API = True
except Exception:
    HAS_API = False


@pytest.mark.skipif(not HAS_API, reason="FastAPI app yüklənmədi")
def test_health_endpoint():
    """GET /health → 200 və düzgün struktur qaytarmalıdır."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.skipif(not HAS_API, reason="FastAPI app yüklənmədi")
def test_predict_endpoint_returns_forecast():
    """GET /predict → proqnoz siyahısı qaytarmalıdır."""
    response = client.get("/predict?horizon_h=6")
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) > 0
    assert "pm25_pred" in data["forecast"][0]


@pytest.mark.skipif(not HAS_API, reason="FastAPI app yüklənmədi")
def test_predict_horizon_respected():
    """horizon_h parametri forecast uzunluğuna uyğun olmalıdır."""
    response = client.get("/predict?horizon_h=12")
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecast"]) == 12


@pytest.mark.skipif(not HAS_API, reason="FastAPI app yüklənmədi")
def test_metrics_endpoint():
    """GET /metrics → model performans göstəriciləri qaytarmalıdır."""
    response = client.get("/metrics")
    assert response.status_code == 200