"""
AirWatch AZ — DuckDB Persistence Layer
=======================================
Replaces CSV storage with an embedded analytical database.
DuckDB is zero-config, file-based, and handles time-series
queries 10-100× faster than pandas+CSV at this scale.

Schema:
  readings  — raw hourly air quality + weather measurements
  forecasts — model predictions with confidence bounds
  model_runs — experiment results per training run
"""

import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DB_DIR

log = logging.getLogger(__name__)
DB_PATH = DB_DIR / "airwatch.duckdb"


# ── Schema DDL ────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS readings (
    timestamp    TIMESTAMPTZ NOT NULL,
    station      VARCHAR     NOT NULL DEFAULT 'baku',
    pm25         DOUBLE,
    pm10         DOUBLE,
    no2          DOUBLE,
    o3           DOUBLE,
    aqi          INTEGER,
    temp         DOUBLE,
    humidity     DOUBLE,
    wind_speed   DOUBLE,
    wind_dir     DOUBLE,
    precip       DOUBLE,
    pressure     DOUBLE,
    is_demo      BOOLEAN     NOT NULL DEFAULT FALSE,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (timestamp, station)
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id  VARCHAR     PRIMARY KEY,
    model_name   VARCHAR     NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    target_time  TIMESTAMPTZ NOT NULL,
    pm25_pred    DOUBLE      NOT NULL,
    pm25_lower   DOUBLE,
    pm25_upper   DOUBLE,
    horizon_h    INTEGER     NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id       VARCHAR     PRIMARY KEY,
    model_name   VARCHAR     NOT NULL,
    trained_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_samples    INTEGER,
    n_features   INTEGER,
    n_splits     INTEGER,
    mae          DOUBLE,
    rmse         DOUBLE,
    mape         DOUBLE,
    r2           DOUBLE,
    mae_std      DOUBLE,
    rmse_std     DOUBLE,
    params       JSON
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def get_conn(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection; creates DB + schema on first use."""
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        conn.execute(_DDL)
    return conn


# ── Write operations ──────────────────────────────────────────────────────────

def upsert_readings(df: pd.DataFrame, is_demo: bool = False) -> int:
    """
    Insert new readings; skip duplicates (timestamp, station).
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    df = df.copy()
    df["is_demo"]     = is_demo
    df["ingested_at"] = datetime.utcnow()
    df["timestamp"]   = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

    if "station" not in df.columns:
        df["station"] = "baku"

    cols = [
        "timestamp", "station", "pm25", "pm10", "no2", "o3", "aqi",
        "temp", "humidity", "wind_speed", "wind_dir", "precip", "pressure",
        "is_demo", "ingested_at",
    ]
    # Only keep columns that exist in the DataFrame
    cols = [c for c in cols if c in df.columns]
    df_insert = df[cols]

    conn = get_conn()
    try:
        before = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        conn.execute(
            f"""
            INSERT OR IGNORE INTO readings ({', '.join(cols)})
            SELECT {', '.join(cols)} FROM df_insert
            """
        )
        after = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        inserted = after - before
        log.info(f"DB upsert: {inserted} new readings (total={after:,})")
        return inserted
    finally:
        conn.close()


def save_forecast(
    model_name: str,
    target_time: datetime,
    pm25_pred: float,
    horizon_h: int,
    pm25_lower: float | None = None,
    pm25_upper: float | None = None,
) -> None:
    """Persist a single prediction row."""
    import uuid
    forecast_id = str(uuid.uuid4())
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO forecasts
                (forecast_id, model_name, target_time, pm25_pred, pm25_lower, pm25_upper, horizon_h)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [forecast_id, model_name, target_time, pm25_pred, pm25_lower, pm25_upper, horizon_h],
        )
    finally:
        conn.close()


def save_model_run(run: dict) -> None:
    """Persist a model training result."""
    import uuid, json
    run_id = str(uuid.uuid4())
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO model_runs
                (run_id, model_name, n_samples, n_features, n_splits,
                 mae, rmse, mape, r2, mae_std, rmse_std, params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                run.get("model"),
                run.get("n_samples"),
                run.get("n_features"),
                run.get("n_splits"),
                run.get("mae"),
                run.get("rmse"),
                run.get("mape"),
                run.get("r2"),
                run.get("mae_std"),
                run.get("rmse_std"),
                json.dumps(run.get("params", {})),
            ],
        )
    finally:
        conn.close()


# ── Read operations ───────────────────────────────────────────────────────────

def load_readings(days: int = 365, station: str = "baku") -> pd.DataFrame:
    """Load readings for the last N days from DuckDB."""
    conn = get_conn(read_only=True)
    try:
        df = conn.execute(
            f"""
            SELECT * FROM readings
            WHERE station = ?
              AND timestamp >= now() - INTERVAL '{days} days'
            ORDER BY timestamp
            """,
            [station],
        ).df()
        log.info(f"DB load: {len(df):,} readings ({days}d, station={station})")
        return df
    finally:
        conn.close()


def load_latest_reading(station: str = "baku") -> dict | None:
    """Return the single most recent reading as a dict."""
    conn = get_conn(read_only=True)
    try:
        row = conn.execute(
            "SELECT * FROM readings WHERE station = ? ORDER BY timestamp DESC LIMIT 1",
            [station],
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in conn.description]
        return dict(zip(cols, row))
    finally:
        conn.close()


def load_forecasts(horizon_h: int = 24) -> pd.DataFrame:
    """Return the latest forecast window."""
    conn = get_conn(read_only=True)
    try:
        return conn.execute(
            """
            SELECT * FROM forecasts
            WHERE predicted_at = (SELECT MAX(predicted_at) FROM forecasts)
              AND horizon_h <= ?
            ORDER BY target_time
            """,
            [horizon_h],
        ).df()
    finally:
        conn.close()


def load_model_runs() -> pd.DataFrame:
    """Return all model training results, newest first."""
    conn = get_conn(read_only=True)
    try:
        return conn.execute(
            "SELECT * FROM model_runs ORDER BY trained_at DESC"
        ).df()
    finally:
        conn.close()


def db_stats() -> dict:
    """Quick health check — row counts per table."""
    conn = get_conn(read_only=True)
    try:
        stats = {}
        for table in ("readings", "forecasts", "model_runs"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        return stats
    finally:
        conn.close()
