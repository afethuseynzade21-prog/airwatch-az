"""
AirWatch AZ — SQLite Database
===============================
Data-nı lokal SQLite database-də saxlayır.

İstifadə:
  from data_source.sql import save_to_db, load_from_db
  save_to_db(df)
  df = load_from_db(days=30)
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = "data/airwatch.db"


def init_db():
    """Database və cədvəl yarat (mövcud deyilsə)."""
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS air_quality (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE,
            pm25      REAL,
            pm10      REAL,
            no2       REAL,
            o3        REAL,
            aqi       REAL,
            temp      REAL,
            humidity  REAL,
            wind_speed REAL,
            station   TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database hazır:", DB_PATH)


def save_to_db(df: pd.DataFrame) -> int:
    """DataFrame-i database-ə saxla."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    cols = ["timestamp", "pm25", "pm10", "no2", "o3", 
            "aqi", "temp", "humidity", "wind_speed", "station"]
    
    existing_cols = [c for c in cols if c in df.columns]
    df_save = df[existing_cols].copy()
    df_save["timestamp"] = pd.to_datetime(df_save["timestamp"]).astype(str)
    
    df_save.to_sql("air_quality", conn, if_exists="append", 
                   index=False)
    
    count = conn.execute("SELECT COUNT(*) FROM air_quality").fetchone()[0]
    conn.close()
    print(f"✅ Database: {count:,} sətir")
    return count


def load_from_db(days: int = 30) -> pd.DataFrame:
    """Database-dən son N günün datasını yüklə."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql(
        f"SELECT * FROM air_quality WHERE timestamp >= '{since}' ORDER BY timestamp",
        conn
    )
    conn.close()
    print(f"✅ Database-dən yükləndi: {len(df):,} sətir")
    return df


if __name__ == "__main__":
    from src.data_pipeline import fetch_all
    df = fetch_all(days=30, save=False)
    save_to_db(df)
    df2 = load_from_db(days=30)
    print(df2.tail())