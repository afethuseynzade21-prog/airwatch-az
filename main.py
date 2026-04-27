"""
AirWatch AZ — Main Entry Point
================================
Bütün pipeline-ı bir əmrlə işə salır:
  python main.py --mode all
  python main.py --mode data
  python main.py --mode train
  python main.py --mode dashboard
"""

import argparse
import os
from dotenv import load_dotenv

# .env faylından token yüklə
load_dotenv()

def run_data():
    print("📡 Data çəkilir...")
    from src.data_pipeline import fetch_all
    df = fetch_all(days=365, save=True)
    print(f"✅ {len(df):,} sətir hazır")
    return df

def run_train():
    print("🤖 Model öyrədilir...")
    from src.data_pipeline import fetch_all
    from src.features import build_features
    from src.train import run_experiment
    df = fetch_all(days=365, save=False)
    X, y, ts = build_features(df)
    results, best, artifacts = run_experiment(X, y, ts, n_splits=5)
    print("✅ Model saxlanıldı: outputs/best_model.pkl")

def run_dashboard():
    print("🌐 Dashboard işə salınır...")
    os.system("streamlit run app/streamlit_app.py")

def main():
    parser = argparse.ArgumentParser(description="AirWatch AZ Pipeline")
    parser.add_argument("--mode", choices=["data", "train", "dashboard", "all"],
                        default="all", help="Hansı hissəni işə sal")
    args = parser.parse_args()

    if args.mode == "data":
        run_data()
    elif args.mode == "train":
        run_train()
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "all":
        run_data()
        run_train()
        run_dashboard()

if __name__ == "__main__":
    main()