"""
AirWatch AZ — LSTM Model (PyTorch)
====================================
Bidirectional LSTM with attention for PM2.5 time-series forecasting.

Architecture:
  Input  → [batch, seq_len, n_features]
  LSTM   → 2-layer bidirectional LSTM (hidden_size=128)
  Attn   → learned attention over time steps
  FC     → single scalar output (PM2.5 t+1)

Training strategy:
  - Sliding window sequences of 24 hours
  - TimeSeriesSplit to prevent lookahead
  - Early stopping on validation MAE
  - Gradient clipping for stability

Install: pip install torch
"""

import logging
import math
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import LSTM_PARAMS

log = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class _PMDataset:
    """Sliding-window dataset; __getitem__ returns (seq, target) tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int):
        import torch
        x_seq = torch.from_numpy(self.X[idx : idx + self.seq_len])
        y_val = torch.tensor(self.y[idx + self.seq_len], dtype=torch.float32)
        return x_seq, y_val


# ── Model definition ──────────────────────────────────────────────────────────

class _LSTMModel:
    """Wrapper so the model fits the sklearn-style .predict() interface."""

    def __init__(self, net, scaler_X: StandardScaler, scaler_y: StandardScaler, seq_len: int):
        self.net      = net
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.seq_len  = seq_len

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on raw (unscaled) feature matrix; returns PM2.5 in μg/m³."""
        import torch
        self.net.eval()
        X_sc = self.scaler_X.transform(X).astype(np.float32)
        preds = []

        # Slide window over test set
        for i in range(self.seq_len, len(X_sc)):
            seq = torch.from_numpy(X_sc[i - self.seq_len : i]).unsqueeze(0)
            with torch.no_grad():
                out = self.net(seq).item()
            preds.append(out)

        # Inverse-transform
        preds_arr = np.array(preds).reshape(-1, 1)
        return self.scaler_y.inverse_transform(preds_arr).ravel()

    def feature_importances_(self) -> None:
        return None   # LSTM has no feature importances; use SHAP instead


def _build_net(n_features: int, hidden: int, layers: int, dropout: float):
    """Build PyTorch LSTM + attention + FC."""
    import torch
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.attn = nn.Linear(hidden * 2, 1)
            self.fc   = nn.Sequential(
                nn.Linear(hidden * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)          # [B, T, 2H]
            # Attention over time steps
            scores = self.attn(out)        # [B, T, 1]
            weights= torch.softmax(scores, dim=1)
            context = (out * weights).sum(dim=1)   # [B, 2H]
            return self.fc(context).squeeze(-1)

    return _Net()


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: dict,
) -> tuple[float, float, float, float]:
    """Train one fold; return (mae, rmse, mape, r2)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install torch")

    seq_len    = params["seq_len"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_X   = StandardScaler().fit(X_tr)
    scaler_y   = StandardScaler().fit(y_tr.reshape(-1, 1))

    X_tr_sc = scaler_X.transform(X_tr).astype(np.float32)
    y_tr_sc = scaler_y.transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32)
    X_te_sc = scaler_X.transform(X_te).astype(np.float32)
    y_te_sc = scaler_y.transform(y_te.reshape(-1, 1)).ravel().astype(np.float32)

    train_ds = _PMDataset(X_tr_sc, y_tr_sc, seq_len)
    train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=False)

    net = _build_net(X_tr.shape[1], params["hidden_size"], params["num_layers"], params["dropout"])
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=params["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    criterion = nn.HuberLoss()

    best_val_loss = math.inf
    patience_left = params["patience"]

    for epoch in range(params["epochs"]):
        net.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()

        # Validation (no teacher forcing — pure autoregressive)
        net.eval()
        val_preds = []
        for i in range(seq_len, len(X_te_sc)):
            seq = torch.from_numpy(X_te_sc[i - seq_len : i]).unsqueeze(0).to(device)
            with torch.no_grad():
                val_preds.append(net(seq).item())

        if not val_preds:
            break

        val_pred_arr = np.array(val_preds)
        val_true_arr = y_te_sc[seq_len:]
        val_loss = float(np.mean(np.abs(val_pred_arr - val_true_arr)))
        sched.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_left = params["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info(f"    Early stop at epoch {epoch+1}")
                break

    # Final evaluation (inverse transform)
    preds_real = scaler_y.inverse_transform(np.array(val_preds).reshape(-1, 1)).ravel()
    true_real  = y_te[seq_len:]

    mae  = float(mean_absolute_error(true_real, preds_real))
    rmse = float(np.sqrt(mean_squared_error(true_real, preds_real)))
    r2   = float(r2_score(true_real, preds_real))
    mask = true_real > 1
    mape = float(np.mean(np.abs((true_real[mask] - preds_real[mask]) / true_real[mask])) * 100) if mask.sum() else 0.0

    return mae, rmse, mape, r2, net, scaler_X, scaler_y


# ── Public API ────────────────────────────────────────────────────────────────

def train_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    tscv: TimeSeriesSplit,
    params: dict | None = None,
) -> tuple[dict, _LSTMModel | None]:
    """
    Train bidirectional LSTM with attention on each TimeSeriesSplit fold.

    Returns:
        (result_dict, trained_model_on_last_fold)
        result_dict keys: model, mae, rmse, mape, r2, mae_std, ...
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("PyTorch not installed — LSTM skipped. Run: pip install torch")
        return None, None

    p = {**LSTM_PARAMS, **(params or {})}
    seq_len = p["seq_len"]

    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    scores = []
    last_model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < seq_len * 2 or len(test_idx) < seq_len + 10:
            log.warning(f"LSTM fold {fold+1}: too few samples — skipping")
            continue

        log.info(f"  LSTM fold {fold+1} | train={len(train_idx)}  test={len(test_idx)}")
        mae, rmse, mape, r2, net, scaler_X, scaler_y = _train_fold(
            X_arr[train_idx], y_arr[train_idx],
            X_arr[test_idx],  y_arr[test_idx],
            p,
        )
        scores.append({"mae": mae, "rmse": rmse, "mape": mape, "r2": r2})
        last_model = _LSTMModel(net, scaler_X, scaler_y, seq_len)
        log.info(f"    → MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    if not scores:
        log.warning("LSTM: no valid folds completed")
        return None, None

    result = {"model": "LSTM"}
    for m in ("mae", "rmse", "mape", "r2"):
        vals = [s[m] for s in scores]
        result[m]          = round(float(np.mean(vals)), 4)
        result[f"{m}_std"] = round(float(np.std(vals)), 4)

    log.info(f"LSTM → MAE={result['mae']:.3f}  R²={result['r2']:.3f}")
    return result, last_model
