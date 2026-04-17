"""
AirWatch AZ — API Pydantic Schemas
====================================
Request / response models for the FastAPI service.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status:      str = "ok"
    version:     str
    model_name:  Optional[str] = None
    db_readings: Optional[int] = None
    timestamp:   datetime


class PredictRequest(BaseModel):
    datetime: Optional[datetime] = Field(
        default=None,
        description="Target datetime for prediction. Defaults to now+1h.",
        examples=["2025-06-15T14:00:00"],
    )
    horizon_h: int = Field(
        default=1,
        ge=1,
        le=72,
        description="Forecast horizon in hours (1–72).",
    )


class RiskLevel(BaseModel):
    label:     str
    risk:      str
    color:     str
    action:    str
    who_ratio: float = Field(description="Multiple of WHO annual guideline (5 μg/m³)")


class ForecastStep(BaseModel):
    target_time: datetime
    horizon_h:   int
    pm25_pred:   float = Field(ge=0, description="Predicted PM2.5 (μg/m³)")
    pm25_lower:  float = Field(ge=0, description="90% prediction interval lower bound")
    pm25_upper:  float = Field(ge=0, description="90% prediction interval upper bound")
    risk_label:  str
    risk_level:  str
    risk_color:  str
    action:      str


class PredictResponse(BaseModel):
    model_name:    str
    generated_at:  datetime
    current:       dict
    forecast:      list[ForecastStep]
    recommendations: dict[str, str]


class MetricsResponse(BaseModel):
    model_name:  str
    trained_at:  Optional[datetime] = None
    n_samples:   Optional[int]      = None
    n_features:  Optional[int]      = None
    mae:         Optional[float]    = Field(default=None, description="Mean Absolute Error (μg/m³)")
    rmse:        Optional[float]    = Field(default=None, description="Root Mean Square Error (μg/m³)")
    mape:        Optional[float]    = Field(default=None, description="Mean Absolute Percentage Error (%)")
    r2:          Optional[float]    = Field(default=None, description="R² coefficient of determination")
    mae_std:     Optional[float]    = None
    rmse_std:    Optional[float]    = None
    leaderboard: list[dict]         = Field(default_factory=list)


class StationReading(BaseModel):
    station:   str
    timestamp: datetime
    pm25:      Optional[float] = None
    aqi:       Optional[int]   = None
    risk_label: str
    risk_level: str


class CurrentResponse(BaseModel):
    stations: list[StationReading]
    generated_at: datetime
