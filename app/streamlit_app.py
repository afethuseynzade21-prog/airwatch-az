"""
AirWatch AZ — Streamlit Dashboard
====================================
3 Tab:
  Tab 1: Risk Monitor  — cari AQI + risk rəngi
  Tab 2: Forecast      — 24h PM2.5 proqnozu
  Tab 3: Alert Logic   — WHO threshold cədvəli

İşə sal:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib

from src.data_pipeline import fetch_all, _get_waqi_token
from src.features import build_features

# Token yoxlaması — istifadəçini məlumatlandır
_token = _get_waqi_token()
if not _token:
    st.warning(
        "⚠️ **WAQI token tapılmadı** — demo data göstərilir.\n\n"
        "Real data üçün: `.streamlit/secrets.toml` faylında `WAQI_TOKEN` əlavə et "
        "və ya Streamlit Cloud → App Settings → Secrets.",
        icon="🔑"
    )

# ── Səhifə konfiqurasiyası ────────────────────────────────────────────────
st.set_page_config(
    page_title="AirWatch AZ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── WHO threshold-lar ─────────────────────────────────────────────────────
WHO_THRESHOLDS = [
    {"label": "Təmiz",     "min": 0,   "max": 12,  "color": "#2ecc71", "action": "Heç bir məhdudiyyət lazım deyil."},
    {"label": "Orta",      "min": 12,  "max": 35,  "color": "#f1c40f", "action": "Həssas qruplar üçün xəbərdarlıq edin."},
    {"label": "Yüksək",    "min": 35,  "max": 55,  "color": "#e67e22", "action": "Trafik məhdudiyyəti tövsiyə edilir. Açıq hava tədbirlərini ləğv edin."},
    {"label": "Kritik",    "min": 55,  "max": 999, "color": "#e74c3c", "action": "Sənaye fəaliyyətini azaldın. Xarici fəaliyyəti dayandırın."},
]


def get_risk(pm25_val: float) -> dict:
    for t in WHO_THRESHOLDS:
        if t["min"] <= pm25_val < t["max"]:
            return t
    return WHO_THRESHOLDS[-1]


# ── Data yüklə (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = fetch_all(days=365, save=False)
    X, y, ts = build_features(df)
    return df, X, y, ts


@st.cache_resource
def load_model_cached():
    model_path = Path("outputs/best_model.pkl")
    if model_path.exists():
        data = joblib.load(model_path)
        return data["model"], data["name"], data["features"]
    return None, None, None


# ════════════════════════════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding: 1rem 0 0.5rem">
    <h1 style="font-size:1.8rem; margin:0">🌿 AirWatch AZ</h1>
    <p style="color:#666; margin:0.2rem 0 0">Bakı Hava Keyfiyyəti Proqnoz &amp; Qərar Dəstəyi Platforması</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Data yüklə ────────────────────────────────────────────────────────────
with st.spinner("Data yüklənir..."):
    df, X, y, ts = load_data()
    model, model_name, feat_cols = load_model_cached()

# ════════════════════════════════════════════════════════════════════════════
# 3 TAB
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📊 Risk Monitor", "🔮 Forecast", "⚡ Alert Logic"])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Risk Monitor
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Cari PM2.5
        latest_pm25 = float(df["pm25"].iloc[-1])
        risk         = get_risk(latest_pm25)
        ts_latest    = df["timestamp"].iloc[-1]

        st.markdown(f"""
        <div style="background:{risk['color']}22; border:2px solid {risk['color']};
                    border-radius:12px; padding:1.2rem; text-align:center; margin-bottom:1rem">
            <p style="margin:0;font-size:0.8rem;color:#555">Son ölçüm · {ts_latest.strftime('%d %b, %H:%M')}</p>
            <p style="margin:0.3rem 0;font-size:3rem;font-weight:700;color:{risk['color']}">{latest_pm25:.0f}</p>
            <p style="margin:0;font-size:0.85rem;color:#444">μg/m³ · PM2.5</p>
            <hr style="margin:0.8rem 0;border-color:{risk['color']}44">
            <p style="margin:0;font-size:1.1rem;font-weight:600;color:{risk['color']}">{risk['label']}</p>
            <p style="margin:0.3rem 0 0;font-size:0.8rem;color:#555">{risk['action']}</p>
        </div>
        """, unsafe_allow_html=True)

        # WHO müqayisəsi
        who_annual = 5.0   # WHO 2021 Annual PM2.5 Guideline
        ratio = latest_pm25 / who_annual
        st.metric(
            "WHO İllik Norması (5 μg/m³)",
            f"{latest_pm25:.0f} μg/m³",
            delta=f"{ratio:.1f}× norma üzərindədir",
            delta_color="inverse",
        )

    with col_right:
        # Son 7 günlük trend
        df_week = df.tail(7 * 24).copy()
        risk_colors = [get_risk(v)["color"] for v in df_week["pm25"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_week["timestamp"],
            y=df_week["pm25"],
            mode="lines",
            line=dict(color="#2c3e50", width=1.5),
            name="PM2.5",
            hovertemplate="<b>%{y:.1f} μg/m³</b><br>%{x}<extra></extra>",
        ))

        # WHO threshold xətləri
        for thresh, label, color in [(12, "WHO Orta", "#f1c40f"), (35, "WHO Yüksək", "#e67e22"), (55, "WHO Kritik", "#e74c3c")]:
            fig.add_hline(
                y=thresh, line_dash="dot", line_color=color, opacity=0.6,
                annotation_text=label, annotation_position="top right",
            )

        fig.update_layout(
            title="Son 7 Günlük PM2.5 Trendi",
            xaxis_title=None,
            yaxis_title="PM2.5 (μg/m³)",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Saatlıq pattern
    st.subheader("Saatlıq Orta PM2.5 Paylanması")
    df["hour"] = df["timestamp"].dt.hour
    hourly_avg  = df.groupby("hour")["pm25"].agg(["mean", "std"]).reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hourly_avg["hour"],
        y=hourly_avg["mean"] + hourly_avg["std"],
        fill=None, mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=hourly_avg["hour"],
        y=hourly_avg["mean"] - hourly_avg["std"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(46,204,113,0.2)", name="±1 std",
    ))
    fig2.add_trace(go.Scatter(
        x=hourly_avg["hour"], y=hourly_avg["mean"],
        mode="lines+markers", line=dict(color="#2ecc71", width=2),
        marker=dict(size=5), name="Orta PM2.5",
    ))
    fig2.update_layout(
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, title="Saat"),
        yaxis_title="PM2.5 (μg/m³)",
        height=280, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Forecast
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    if model is None:
        st.warning("Model tapılmadı. Əvvəlcə `python -m src.train` işlət.")
    else:
        st.caption(f"Aktiv model: **{model_name}**")

        # Son 48 saatın actual vs predicted
        n_pred = min(48, len(X))
        X_recent = X.iloc[-n_pred:]
        y_recent = y.iloc[-n_pred:]
        ts_recent = ts.iloc[-n_pred:]

        if feat_cols:
            # Mövcud feature-lara uyğunlaşdır
            common = [c for c in feat_cols if c in X_recent.columns]
            preds  = model.predict(X_recent[common])
        else:
            preds = model.predict(X_recent)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=ts_recent, y=y_recent.values,
            mode="lines", name="Actual",
            line=dict(color="#2c3e50", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=ts_recent, y=preds,
            mode="lines", name="Proqnoz",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        ))
        fig3.update_layout(
            title="Actual vs Proqnoz (Son 48 Saat)",
            xaxis_title=None, yaxis_title="PM2.5 (μg/m³)",
            height=340, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance (Əhəmiyyət Sıralaması)")
            feats = feat_cols or list(X.columns)
            imp_df = (
                pd.DataFrame({"feature": feats[:len(model.feature_importances_)],
                              "importance": model.feature_importances_})
                .sort_values("importance", ascending=True)
                .tail(15)
            )
            fig4 = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Greens",
            )
            fig4.update_layout(
                height=400, margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig4, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Alert Logic
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("WHO PM2.5 Threshold → Konkret Tədbirlər")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Interaktiv PM2.5 girişi
        pm25_input = st.slider(
            "PM2.5 dəyəri daxil et (μg/m³)",
            min_value=0.0, max_value=150.0,
            value=float(latest_pm25), step=0.5,
        )
        risk_selected = get_risk(pm25_input)
        st.markdown(f"""
        <div style="background:{risk_selected['color']}22;
                    border-left:4px solid {risk_selected['color']};
                    border-radius:0 8px 8px 0; padding:1rem; margin-top:1rem">
            <b style="color:{risk_selected['color']};font-size:1.1rem">{risk_selected['label']}</b>
            <p style="margin:0.5rem 0 0;font-size:0.9rem">{risk_selected['action']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Alert cədvəli
        alert_df = pd.DataFrame([
            {"Səviyyə": t["label"], "PM2.5 (μg/m³)": f"{t['min']}–{t['max'] if t['max']<999 else '∞'}",
             "Tətbiqçi": ("Bələdiyyə" if t["min"]==0
                          else "Məktəblər" if t["min"]==12
                          else "Bələdiyyə · Səhiyyə" if t["min"]==35
                          else "Nazirlik · SOCAR"),
             "Tövsiyə": t["action"]}
            for t in WHO_THRESHOLDS
        ])
        st.dataframe(alert_df, use_container_width=True, hide_index=True)

    st.divider()

    # Xəbərdarlıq
    st.markdown("""
    <div style="background:#fff3cd;border-left:4px solid #ffc107;
                border-radius:0 8px 8px 0;padding:0.85rem 1rem;margin-top:0.5rem">
        <b>⚠️ Causal Xəbərdarlıq:</b> Bu model PM2.5 ilə hava şəraiti arasında
        <em>statistik korrelyasiyaları</em> müəyyən edir. Sübut edilmiş <em>səbəb-nəticə</em>
        əlaqəsi deyil. Siyasi qərarlar üçün sahə ekspertlərinin təsdiqi lazımdır.
    </div>
    """, unsafe_allow_html=True)

    # Son statistika
    st.divider()
    st.subheader("Son 30 Günün Statistikası")
    df_30 = df.tail(30 * 24)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Orta PM2.5",   f"{df_30['pm25'].mean():.1f} μg/m³")
    mcol2.metric("Maksimum",     f"{df_30['pm25'].max():.1f} μg/m³")
    mcol3.metric("Kritik Saat",  f"{(df_30['pm25']>55).sum()} saat")
    mcol4.metric("WHO Norma Keçmə", f"{(df_30['pm25']>12).mean()*100:.0f}%")


# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AirWatch AZ · Data: WAQI API + Open-Meteo · "
    "Model: correlation-based predictive, not causal · "
    f"Son yeniləmə: {datetime.now().strftime('%d %b %Y, %H:%M')}"
)
