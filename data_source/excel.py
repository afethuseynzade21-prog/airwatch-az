"""
AirWatch AZ — Excel Export
===========================
Data və model nəticələrini Excel-ə export edir.

İstifadə:
  from data_source.excel import export_to_excel
  export_to_excel(df, "outputs/report.xlsx")
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def export_to_excel(df: pd.DataFrame, output_path: str = None) -> str:
    """
    DataFrame-i Excel faylına export et.
    
    Args:
        df: export ediləcək data
        output_path: fayl yolu (default: outputs/airwatch_YYYYMMDD.xlsx)
    
    Returns:
        Excel fayl yolu
    """
    if output_path is None:
        Path("outputs").mkdir(exist_ok=True)
        output_path = f"outputs/airwatch_{datetime.now().strftime('%Y%m%d')}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Ana data
        df.to_excel(writer, sheet_name="PM2.5 Data", index=False)

        # Günlük xülasə
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date")["pm25"].agg(
            ["mean", "max", "min", "std"]
        ).round(2)
        daily.columns = ["Orta", "Maks", "Min", "Std"]
        daily.to_excel(writer, sheet_name="Günlük Xülasə")

        # WHO threshold keçmələri
        who_exceed = df[df["pm25"] > 15][["timestamp", "pm25"]].copy()
        who_exceed.to_excel(writer, sheet_name="WHO Keçmələri", index=False)

    print(f"✅ Excel saxlanıldı: {output_path}")
    return output_path


if __name__ == "__main__":
    from src.data_pipeline import fetch_all
    df = fetch_all(days=30, save=False)
    export_to_excel(df)