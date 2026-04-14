"""
AirWatch AZ — Model Training
==============================
TimeSeriesSplit validation ilə 3 model müqayisəsi:
  1. Persistence Baseline
  2. Random Forest
  3. LightGBM (Phase 2)

İstifadə:
    from src.train import run_experiment
    results, best_model = run_experiment(X, y)
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import joblib
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)
MODEL_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. Metrik hesablama
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, MAPE, R² hesabla."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — sıfır dəyərlərdən qoru
    mask = y_true > 1
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ════════════════════════════════════════════════════════════════════════════
# 2. Baseline model
# ════════════════════════════════════════════════════════════════════════════

def persistence_baseline(X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> dict:
    """
    Persistence model: "dünənki dəyəri saxla" (pm25_lag_1h).
    Bu ən sadə baseline-dır. Hər model bundan yaxşı olmalıdır.
    """
    scores = []
    for train_idx, test_idx in tscv.split(X):
        y_test = y.iloc[test_idx].values
        # pm25_lag_1h = shift(1) — yəni 1 saat əvvəlki dəyər
        y_pred = X["pm25_lag_1h"].iloc[test_idx].values
        scores.append(compute_metrics(y_test, y_pred))

    return _aggregate_scores(scores, "Persistence (baseline)")


# ════════════════════════════════════════════════════════════════════════════
# 3. Ridge Regression
# ════════════════════════════════════════════════════════════════════════════

def train_ridge(X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> tuple[dict, object]:
    """Ridge regression — sadə, interpretable baseline."""
    scaler = StandardScaler()
    scores = []
    model  = None

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_sc, y_tr)
        y_pred = model.predict(X_te_sc)

        scores.append(compute_metrics(y_te, y_pred))

    # Son fold modeli saxla
    scaler_final = StandardScaler().fit(X)
    model_final  = Ridge(alpha=1.0).fit(scaler_final.transform(X), y)

    return _aggregate_scores(scores, "Ridge Regression"), model_final


# ════════════════════════════════════════════════════════════════════════════
# 4. Random Forest
# ════════════════════════════════════════════════════════════════════════════

def train_random_forest(X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> tuple[dict, object]:
    """Random Forest — Phase 1 əsas modeli."""
    scores = []
    model  = None

    rf_params = {
        "n_estimators":       200,
        "max_depth":          12,
        "min_samples_leaf":   5,
        "max_features":       "sqrt",
        "n_jobs":             -1,
        "random_state":       42,
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        model = RandomForestRegressor(**rf_params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        m = compute_metrics(y_te, y_pred)
        scores.append(m)
        log.info(f"  RF Fold {fold+1}: MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R²={m['r2']:.3f}")

    # Tam dataset ilə final model train et
    model_final = RandomForestRegressor(**rf_params).fit(X, y)

    return _aggregate_scores(scores, "Random Forest"), model_final


# ════════════════════════════════════════════════════════════════════════════
# 5. LightGBM (Phase 2)
# ════════════════════════════════════════════════════════════════════════════

def train_lightgbm(X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> tuple[dict, object]:
    """
    LightGBM — Phase 2 əsas modeli.
    lightgbm paketi qurulmalıdır: pip install lightgbm
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LightGBM qurulmayıb. pip install lightgbm")
        return None, None

    scores = []
    model  = None

    lgb_params = {
        "n_estimators":       500,
        "learning_rate":      0.05,
        "max_depth":          8,
        "num_leaves":         31,
        "min_child_samples":  20,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "reg_alpha":          0.1,
        "reg_lambda":         0.1,
        "n_jobs":             -1,
        "random_state":       42,
        "verbose":            -1,
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        y_pred = model.predict(X_te)

        m = compute_metrics(y_te, y_pred)
        scores.append(m)
        log.info(f"  LGB Fold {fold+1}: MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R²={m['r2']:.3f}")

    model_final = lgb.LGBMRegressor(**lgb_params).fit(X, y)

    return _aggregate_scores(scores, "LightGBM"), model_final


# ════════════════════════════════════════════════════════════════════════════
# 6. Əsas experiment funksiyası
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    save_model: bool = True,
) -> tuple[pd.DataFrame, object]:
    """
    3 modeli TimeSeriesSplit ilə müqayisə et.

    Args:
        X: feature matrix (build_features() çıxışı)
        y: hədəf sütun
        n_splits: TimeSeriesSplit fold sayı
        save_model: ən yaxşı modeli disk-ə saxla

    Returns:
        results_df: model müqayisə cədvəli
        best_model: ən yaxşı model obyekti
    """
    log.info("=" * 50)
    log.info("Model Experiment Başlayır")
    log.info(f"Dataset: {len(X):,} sətir × {X.shape[1]} feature")
    log.info(f"TimeSeriesSplit: {n_splits} fold")
    log.info("=" * 50)

    tscv    = TimeSeriesSplit(n_splits=n_splits)
    results = []
    models  = {}

    # ── Baseline ────────────────────────────────────────────────────────
    log.info("\n[1/3] Persistence Baseline...")
    scores_base = persistence_baseline(X, y, tscv)
    results.append(scores_base)

    # ── Ridge ───────────────────────────────────────────────────────────
    log.info("\n[2/3] Ridge Regression...")
    scores_ridge, model_ridge = train_ridge(X, y, tscv)
    results.append(scores_ridge)
    models["Ridge"] = model_ridge

    # ── Random Forest ───────────────────────────────────────────────────
    log.info("\n[3/3] Random Forest...")
    scores_rf, model_rf = train_random_forest(X, y, tscv)
    results.append(scores_rf)
    models["RandomForest"] = model_rf

    # ── LightGBM (əgər qurulubsa) ───────────────────────────────────────
    try:
        import lightgbm  # noqa
        log.info("\n[4/4] LightGBM...")
        scores_lgb, model_lgb = train_lightgbm(X, y, tscv)
        if scores_lgb:
            results.append(scores_lgb)
            models["LightGBM"] = model_lgb
    except ImportError:
        log.info("LightGBM keçildi (qurulmayıb)")

    # ── Nəticə cədvəli ──────────────────────────────────────────────────
    df_results = pd.DataFrame(results).set_index("model")
    df_results = df_results.sort_values("mae")

    print("\n" + "=" * 65)
    print("  MODEL MÜQAYİSƏ CƏDVƏLİ (TimeSeriesSplit, n_splits={})".format(n_splits))
    print("=" * 65)
    print(df_results.to_string())
    print("=" * 65)
    print("  * MAE/RMSE: μg/m³ ilə  |  MAPE: %  |  R²: 0–1 arası")
    print("  ⚠️  Bu dəyərlər ± std deviation — ortalamadır")
    print("=" * 65)

    # ── Ən yaxşı model ──────────────────────────────────────────────────
    best_name  = df_results["mae"].idxmin()
    best_model = models.get(best_name)
    log.info(f"\nƏn yaxşı model: {best_name} (MAE={df_results.loc[best_name,'mae']:.2f})")

    # ── Model saxla ─────────────────────────────────────────────────────
    if save_model and best_model:
        model_path = MODEL_DIR / "best_model.pkl"
        joblib.dump({"model": best_model, "name": best_name, "features": list(X.columns)}, model_path)
        log.info(f"Model saxlanıldı: {model_path}")

        # Nəticəni JSON-a yaz (README üçün)
        results_path = MODEL_DIR / "results.json"
        results_path.write_text(
            json.dumps(df_results.reset_index().to_dict(orient="records"), indent=2)
        )
        log.info(f"Nəticələr saxlanıldı: {results_path}")

    return df_results, best_model


# ════════════════════════════════════════════════════════════════════════════
# Köməkçi funksiyalar
# ════════════════════════════════════════════════════════════════════════════

def _aggregate_scores(scores: list[dict], model_name: str) -> dict:
    """Bütün fold nəticələrini ortalama ± std kimi topla."""
    metrics = ["mae", "rmse", "mape", "r2"]
    result  = {"model": model_name}
    for m in metrics:
        vals = [s[m] for s in scores]
        result[m]           = round(np.mean(vals), 3)
        result[f"{m}_std"]  = round(np.std(vals), 3)
    return result


def load_model(path: str = "outputs/best_model.pkl") -> tuple[object, str, list]:
    """Saxlanılmış modeli yüklə."""
    data = joblib.load(path)
    return data["model"], data["name"], data["features"]


if __name__ == "__main__":
    # Test — demo data ilə
    from src.data_pipeline import fetch_all
    from src.features import build_features

    df = fetch_all(days=180)
    X, y, ts = build_features(df)
    results, best = run_experiment(X, y, n_splits=5)
