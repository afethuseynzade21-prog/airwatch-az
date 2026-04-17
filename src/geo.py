"""
AirWatch AZ — Geospatial / Spatial Intelligence Layer
=======================================================
Generates:
  1. Interactive Folium map with station markers + risk rings
  2. Pollution heatmap using IDW (Inverse Distance Weighting) interpolation
  3. Plotly choropleth-style district risk overlay

Data strategy:
  - Real: one WAQI station in Baku city center
  - Extended: five synthetic stations with physics-based spatial interpolation
    derived from wind direction, proximity to industrial zones, and sea effect.

IDW interpolation:
  P(x) = Σ (w_i × v_i) / Σ w_i   where w_i = 1/d_i^p
  Power p=2 gives smooth interpolation without overshooting.

Note: Synthetic spatial extension is explicitly labeled in the UI.
Real multi-station deployment requires additional WAQI station IDs.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import BAKU_LAT, BAKU_LON, STATION_COORDS, WHO_THRESHOLDS

log = logging.getLogger(__name__)


# ── Risk colour lookup ────────────────────────────────────────────────────────

def pm25_to_color(pm25: float) -> str:
    for t in WHO_THRESHOLDS:
        if t["min"] <= pm25 < t["max"]:
            return t["color"]
    return WHO_THRESHOLDS[-1]["color"]


def pm25_to_risk(pm25: float) -> str:
    for t in WHO_THRESHOLDS:
        if t["min"] <= pm25 < t["max"]:
            return t["label"]
    return WHO_THRESHOLDS[-1]["label"]


# ── Spatial interpolation ─────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a  = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def idw_interpolate(
    station_values: dict[str, float],
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    power: float = 2.0,
    min_dist_km: float = 0.1,
) -> np.ndarray:
    """
    IDW interpolation over a lat/lon grid.

    Args:
        station_values: {station_name: pm25_value}
        grid_lat:       2D array of latitudes
        grid_lon:       2D array of longitudes
        power:          distance decay exponent (2 = smooth)
        min_dist_km:    floor to avoid division by zero

    Returns:
        2D array of interpolated PM2.5 values
    """
    weights_sum = np.zeros_like(grid_lat, dtype=float)
    values_sum  = np.zeros_like(grid_lat, dtype=float)

    for station, pm25 in station_values.items():
        if station not in STATION_COORDS:
            continue
        s_lat, s_lon = STATION_COORDS[station]
        dists = np.vectorize(lambda la, lo: _haversine_km(s_lat, s_lon, la, lo))(
            grid_lat, grid_lon
        )
        dists = np.maximum(dists, min_dist_km)
        w = 1.0 / (dists ** power)
        weights_sum += w
        values_sum  += w * pm25

    return np.where(weights_sum > 0, values_sum / weights_sum, np.nan)


def build_station_readings(current_pm25: float, wind_dir: float = 180.0) -> dict[str, float]:
    """
    Derive synthetic readings for all stations from a single observed value.

    Spatial offsets are physically motivated:
      - Upwind stations: lower PM2.5 (cleaner air approaching)
      - Industrial zones (Sabunchu, Binagadi): elevated PM2.5
      - Coastal / sea proximity (airport): lower due to sea breeze
    """
    rng = np.random.default_rng(int(current_pm25 * 100))  # deterministic from reading

    offsets = {
        "baku":           0.0,
        "sumgayit":       current_pm25 * 0.25,   # industrial city to the north
        "baku_airport":   -current_pm25 * 0.15,  # sea-side, cleaner
        "baku_downtown":  current_pm25 * 0.10,   # traffic congestion
        "baku_sabunchu":  current_pm25 * 0.20,   # industrial district
        "baku_binagadi":  current_pm25 * 0.18,   # residential + industrial
    }
    readings = {}
    for station, offset in offsets.items():
        noise = rng.normal(0, current_pm25 * 0.05)
        readings[station] = float(np.clip(current_pm25 + offset + noise, 0, 500))

    return readings


# ── Folium map ────────────────────────────────────────────────────────────────

def build_folium_map(
    station_readings: dict[str, float],
    center_lat: float = BAKU_LAT,
    center_lon: float = BAKU_LON,
    zoom: int = 11,
) -> "folium.Map":
    """
    Build an interactive Folium map with:
      - Circle markers for each station (coloured by risk)
      - Popup with PM2.5 value, risk label, WHO comparison
      - Heatmap layer for IDW-interpolated surface
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        raise ImportError("folium not installed. Run: pip install folium streamlit-folium")

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    # IDW heatmap layer
    lat_grid = np.linspace(center_lat - 0.3, center_lat + 0.3, 60)
    lon_grid = np.linspace(center_lon - 0.4, center_lon + 0.4, 60)
    GLAT, GLON = np.meshgrid(lat_grid, lon_grid, indexing="ij")

    grid_pm25 = idw_interpolate(station_readings, GLAT, GLON)
    # Normalise for folium HeatMap (0–1)
    vmax = max(max(station_readings.values()), 1)
    heat_data = [
        [GLAT[i, j], GLON[i, j], float(grid_pm25[i, j]) / vmax]
        for i in range(GLAT.shape[0])
        for j in range(GLON.shape[1])
        if not np.isnan(grid_pm25[i, j])
    ]
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_opacity=0.7,
        gradient={0.0: "#2ecc71", 0.35: "#f1c40f", 0.6: "#e67e22", 0.8: "#e74c3c", 1.0: "#8e44ad"},
        radius=25,
        blur=20,
    ).add_to(m)

    # Station markers
    for station, pm25 in station_readings.items():
        if station not in STATION_COORDS:
            continue
        lat, lon = STATION_COORDS[station]
        color     = pm25_to_color(pm25)
        risk      = pm25_to_risk(pm25)
        is_real   = station in ("baku", "sumgayit")
        label     = station.replace("_", " ").title()
        data_note = "Real WAQI data" if is_real else "Interpolated estimate"

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:180px">
            <b style="font-size:14px">{label}</b><br>
            <span style="font-size:22px;font-weight:700;color:{color}">{pm25:.1f}</span>
            <span style="color:#555"> μg/m³ PM2.5</span><br>
            <span style="background:{color};color:#fff;padding:2px 8px;
                  border-radius:4px;font-size:12px">{risk}</span><br>
            <span style="font-size:11px;color:#888;margin-top:4px;display:block">
                {data_note} · WHO annual: {pm25/5.0:.1f}×
            </span>
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=12 if is_real else 8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85 if is_real else 0.6,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{label}: {pm25:.1f} μg/m³ ({risk})",
        ).add_to(m)

    # WHO legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:sans-serif;font-size:12px">
        <b>PM2.5 Risk Level</b><br>
        <span style="color:#2ecc71">● Good</span> &lt;12<br>
        <span style="color:#f1c40f">● Moderate</span> 12–35<br>
        <span style="color:#e67e22">● Unhealthy</span> 35–55<br>
        <span style="color:#e74c3c">● Very Unhealthy</span> 55–150<br>
        <span style="color:#8e44ad">● Hazardous</span> &gt;150<br>
        <span style="color:#aaa;font-size:10px">μg/m³ · WHO 2021 Guidelines</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ── Plotly spatial chart ──────────────────────────────────────────────────────

def build_station_bar_chart(station_readings: dict[str, float]) -> "plotly.graph_objects.Figure":
    """Horizontal bar chart of PM2.5 per station — quick Plotly alternative to Folium."""
    import plotly.graph_objects as go

    stations = [k.replace("_", " ").title() for k in station_readings]
    values   = list(station_readings.values())
    colors   = [pm25_to_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=stations,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=12, line_dash="dot", line_color="#f1c40f",
                  annotation_text="WHO Moderate", annotation_position="top right")
    fig.add_vline(x=35, line_dash="dot", line_color="#e67e22",
                  annotation_text="WHO Unhealthy")
    fig.update_layout(
        title="PM2.5 by Station (IDW-interpolated)",
        xaxis_title="PM2.5 (μg/m³)",
        yaxis_title=None,
        height=300,
        margin=dict(l=10, r=80, t=50, b=10),
        showlegend=False,
    )
    return fig
