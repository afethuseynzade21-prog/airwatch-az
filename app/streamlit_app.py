"""
AirWatch AZ — İstehsal Streamlit Paneli v2
==========================================
5 Tab:
  1. Risk Monitoru   — cari AKİ, 7 günlük trend, ÜST müqayisəsi
  2. Proqnoz         — 24 saatlıq təxmin, etibarlılıq intervalları
  3. Coğrafi Xəritə  — interaktiv istilik xəritəsi + stansiya cədvəli
  4. Model Laboratoriyası — liderlik cədvəli, SHAP əhəmiyyəti, xəta analizi
  5. Biznes Mərkəzi  — sağlamlıq/siyasət tövsiyələri, ESG məlumatları

İşə salma:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

from src.config import WHO_THRESHOLDS, MODEL_DIR
from src.data_pipeline import fetch_all
from src.features import build_features
from src.inference import classify_risk, classify_risk_series, get_recommendations

# ── Səhifə konfiqurasiyası ────────────────────────────────────────────────────

st.set_page_config(
    page_title="AirWatch AZ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa; border-radius: 12px; padding: 1.1rem;
    border: 1px solid #e9ecef; margin-bottom: 0.5rem;
  }
  .risk-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; color: #fff;
  }
  .section-title { font-size: 1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ── Məlumat yükləmə (keşlənmiş) ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Hava keyfiyyəti məlumatları yüklənir...")
def load_data(days: int = 365):
    df       = fetch_all(days=days, save=False)
    X, y, ts = build_features(df)
    return df, X, y, ts


@st.cache_resource(show_spinner="Model yüklənir...")
def load_model():
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        return None, None, None
    d = joblib.load(path)
    return d["model"], d.get("name", "Naməlum"), d.get("features", [])


@st.cache_data(ttl=600)
def load_results():
    path = MODEL_DIR / "results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@st.cache_data(ttl=600)
def load_shap():
    path = MODEL_DIR / "shap_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def load_errors():
    path = MODEL_DIR / "error_analysis.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Başlıq ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;padding:0.5rem 0 0.2rem">
  <span style="font-size:2rem">🌿</span>
  <div>
    <h1 style="margin:0;font-size:1.6rem;font-weight:700">AirWatch AZ</h1>
    <p style="margin:0;color:#6c757d;font-size:0.85rem">
      Bakı Hava Keyfiyyəti Kəşfiyyat Platforması · WAQI + Open-Meteo · ML ilə idarə olunur
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Məlumat yüklə ─────────────────────────────────────────────────────────────

df, X, y, ts = load_data()
model, model_name, feat_cols = load_model()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Monitoru", "🔮 Proqnoz", "🗺️ Coğrafi Xəritə", "🧪 Model Laboratoriyası", "💼 Biznes Mərkəzi"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Monitoru
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    latest_pm25 = float(df["pm25"].iloc[-1])
    risk        = classify_risk(latest_pm25)
    ts_latest   = df["timestamp"].iloc[-1]

    col_left, col_mid, col_right = st.columns([1, 1, 2])

    with col_left:
        st.markdown(f"""
        <div style="background:{risk['color']}18;border:2px solid {risk['color']};
                    border-radius:14px;padding:1.4rem;text-align:center">
          <p style="margin:0;font-size:0.75rem;color:#6c757d">
            Son ölçüm · {ts_latest.strftime('%d %b %Y, %H:%M')}
          </p>
          <p style="margin:0.4rem 0;font-size:3.2rem;font-weight:800;
                    color:{risk['color']};line-height:1">{latest_pm25:.1f}</p>
          <p style="margin:0;font-size:0.8rem;color:#555">μg/m³ · PM2.5</p>
          <hr style="border-color:{risk['color']}44;margin:0.8rem 0">
          <span class="risk-badge" style="background:{risk['color']}">{risk['label']}</span>
          <p style="margin:0.5rem 0 0;font-size:0.78rem;color:#444">{risk['action']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_mid:
        who_ratio = latest_pm25 / 5.0
        st.metric("ÜST İllik Hədd (5 μg/m³)", f"{latest_pm25:.1f} μg/m³",
                  delta=f"{who_ratio:.1f}× həddən yuxarı", delta_color="inverse")
        st.metric("24 Saatlıq Ortalama", f"{df['pm25'].tail(24).mean():.1f} μg/m³")
        st.metric("7 Günlük Ortalama", f"{df['pm25'].tail(168).mean():.1f} μg/m³")
        exceedances = int((df["pm25"].tail(168) > 35).sum())
        st.metric("ÜST Zərərli Saatlar (7 gün)", f"{exceedances} saat")

    with col_right:
        df_week = df.tail(7 * 24).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_week["timestamp"], y=df_week["pm25"],
            mode="lines", line=dict(color="#3498db", width=1.5),
            name="PM2.5", fill="tozeroy",
            fillcolor="rgba(52,152,219,0.1)",
            hovertemplate="<b>%{y:.1f} μg/m³</b><br>%{x}<extra></extra>",
        ))
        for thresh, label, color in [
            (12, "Orta", "#f1c40f"),
            (35, "Zərərli", "#e67e22"),
            (55, "Çox Zərərli", "#e74c3c"),
        ]:
            fig.add_hline(y=thresh, line_dash="dot", line_color=color, opacity=0.7,
                          annotation_text=label, annotation_position="top right")
        fig.update_layout(
            title="7 Günlük PM2.5 Trendi",
            yaxis_title="PM2.5 (μg/m³)",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Saatlıq nümunə
    st.subheader("Saatlıq Nümunə (bütün məlumatlar)")
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour")["pm25"].agg(["mean", "std", "median"]).reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hourly["hour"], y=(hourly["mean"] + hourly["std"]),
        fill=None, mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=hourly["hour"], y=(hourly["mean"] - hourly["std"]).clip(lower=0),
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(52,152,219,0.15)", name="±1 std",
    ))
    fig2.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean"], mode="lines+markers",
        line=dict(color="#3498db", width=2.5), marker=dict(size=5), name="Ortalama PM2.5",
    ))
    fig2.update_layout(
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, title="Günün Saatı"),
        yaxis_title="PM2.5 (μg/m³)",
        height=260, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Aylıq mövsümilik
    df["month"] = df["timestamp"].dt.month
    monthly = df.groupby("month")["pm25"].mean().reset_index()
    monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
    fig3 = px.bar(
        monthly, x="month_name", y="pm25", color="pm25",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        title="Aylıq Ortalama PM2.5 (Mövsümi Nümunə)",
        labels={"pm25": "μg/m³", "month_name": "Ay"},
    )
    fig3.add_hline(y=12, line_dash="dot", line_color="#aaa", annotation_text="ÜST Orta Hədd")
    fig3.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10),
                       coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Proqnoz
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if model is None:
        st.warning("Öyrədilmiş model tapılmadı. Əmri icra edin: `python -m src.train`")
        st.stop()

    st.caption(f"Aktiv model: **{model_name}** · {len(feat_cols)} xüsusiyyət")

    horizon = st.slider("Proqnoz müddəti (saat)", 6, 48, 24, step=6)

    with st.spinner("Proqnoz hazırlanır..."):
        try:
            from src.inference import forecast_horizon
            fc_df = forecast_horizon(model, model_name, feat_cols, df.tail(200), horizon)
        except Exception as exc:
            st.error(f"Proqnoz uğursuz oldu: {exc}")
            fc_df = pd.DataFrame()

    if not fc_df.empty:
        col_fc1, col_fc2 = st.columns([3, 1])

        with col_fc1:
            n_hist = min(48, len(df))
            df_hist = df.tail(n_hist)[["timestamp", "pm25"]].copy()
            df_hist["type"] = "Faktiki"

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["pm25"],
                mode="lines", name="Faktiki",
                line=dict(color="#2c3e50", width=2),
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["target_time"], y=fc_df["pm25_upper"],
                fill=None, mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["target_time"], y=fc_df["pm25_lower"],
                fill="tonexty", mode="lines", line=dict(width=0),
                fillcolor="rgba(231,76,60,0.15)", name="90% interval",
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["target_time"], y=fc_df["pm25_pred"],
                mode="lines+markers", name=f"{model_name} proqnozu",
                line=dict(color="#e74c3c", width=2.5, dash="dash"),
                marker=dict(size=4),
            ))
            for thresh, color in [(12, "#f1c40f"), (35, "#e67e22"), (55, "#e74c3c")]:
                fig_fc.add_hline(y=thresh, line_dash="dot", line_color=color, opacity=0.5)

            fig_fc.update_layout(
                title=f"PM2.5 Proqnozu — Növbəti {horizon} saat",
                yaxis_title="PM2.5 (μg/m³)", xaxis_title=None,
                height=360, margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        with col_fc2:
            st.markdown("**Proqnoz Risk Xülasəsi**")
            risk_counts = fc_df["risk_label"].value_counts()
            for label, count in risk_counts.items():
                color = next((t["color"] for t in WHO_THRESHOLDS if t["label"] == label), "#999")
                pct   = count / len(fc_df) * 100
                st.markdown(
                    f'<span class="risk-badge" style="background:{color}">{label}</span>'
                    f' {pct:.0f}% ({count} saat)',
                    unsafe_allow_html=True,
                )
            st.divider()
            peak_row = fc_df.loc[fc_df["pm25_pred"].idxmax()]
            st.metric("Pik Proqnoz", f"{peak_row['pm25_pred']:.1f} μg/m³",
                      delta=f"+{peak_row['horizon_h']} saat")
            min_row = fc_df.loc[fc_df["pm25_pred"].idxmin()]
            st.metric("Ən Yaxşı Pəncərə", f"{min_row['pm25_pred']:.1f} μg/m³",
                      delta=f"+{min_row['horizon_h']} saat", delta_color="inverse")

        with st.expander("Tam proqnoz cədvəli"):
            st.dataframe(
                fc_df[["target_time", "pm25_pred", "pm25_lower", "pm25_upper", "risk_label"]].rename(
                    columns={"target_time": "Vaxt", "pm25_pred": "PM2.5", "pm25_lower": "Aşağı",
                             "pm25_upper": "Yuxarı", "risk_label": "Risk"}
                ).style.format({"PM2.5": "{:.2f}", "Aşağı": "{:.2f}", "Yuxarı": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Coğrafi Xəritə
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Bakı Hava Keyfiyyətinin Məkan Paylanması")
    st.caption(
        "Bir real WAQI stansiyası (Bakı mərkəzi) + Tərs Məsafə Çəkisi (IDW) üsulu ilə "
        "beş interpolyasiya edilmiş təxmin. Süni stansiyalar açıq şəkildə işarələnib."
    )

    try:
        from src.geo import build_folium_map, build_station_readings, build_station_bar_chart
        from streamlit_folium import st_folium

        latest_pm25 = float(df["pm25"].iloc[-1])
        wind_dir    = float(df["wind_dir"].iloc[-1]) if "wind_dir" in df.columns else 180.0
        station_readings = build_station_readings(latest_pm25, wind_dir)

        col_map, col_bar = st.columns([2, 1])

        with col_map:
            from src.geo_simple import show_map
            show_map()

        with col_bar:
            st.plotly_chart(
                build_station_bar_chart(station_readings),
                use_container_width=True,
            )

            st.markdown("**Stansiya Təfərrüatları**")
            for station, pm25 in station_readings.items():
                from src.geo import pm25_to_color, pm25_to_risk
                color = pm25_to_color(pm25)
                risk  = pm25_to_risk(pm25)
                label = station.replace("_", " ").title()
                is_real = station in ("baku", "sumgayit")
                st.markdown(
                    f'<span class="risk-badge" style="background:{color}">{risk}</span> '
                    f'**{label}** — {pm25:.1f} μg/m³'
                    + (' ✓' if is_real else ' *(təxmini)*'),
                    unsafe_allow_html=True,
                )

    except ImportError as exc:
        st.info(f"Xəritə əlavə paketlər tələb edir: `pip install folium streamlit-folium`\n\n{exc}")
    except Exception as exc:
        st.error(f"Xəritə xətası: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Laboratoriyası
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    col_lb, col_shap = st.columns([1, 1])

    with col_lb:
        st.subheader("Model Liderlik Cədvəli")
        results = load_results()
        if results:
            df_lb = pd.DataFrame(results)
            if "model" in df_lb.columns:
                df_lb = df_lb.set_index("model")
            show_cols = ["mae", "mae_std", "rmse", "r2", "mape"]
            show_cols = [c for c in show_cols if c in df_lb.columns]
            st.dataframe(
                df_lb[show_cols].round(3).style.background_gradient(
                    subset=["mae", "rmse"], cmap="RdYlGn_r"
                ).background_gradient(subset=["r2"], cmap="RdYlGn"),
                use_container_width=True,
            )
            st.caption("MAE / RMSE μg/m³ ilə (aşağı daha yaxşı) · R² (yuxarı daha yaxşı)")
        else:
            st.info("Model nəticələrini yaratmaq üçün `python -m src.train` əmrini icra edin.")

    with col_shap:
        st.subheader("SHAP Xüsusiyyət Əhəmiyyəti")
        shap_df = load_shap()
        if not shap_df.empty:
            shap_top = shap_df.head(15).sort_values("shap_abs_mean")
            fig_shap = go.Figure(go.Bar(
                x=shap_top["shap_abs_mean"], y=shap_top["feature"],
                orientation="h",
                marker_color=px.colors.sequential.Blues[3:],
            ))
            fig_shap.update_layout(
                title="Ortalama |SHAP| — Xüsusiyyət Təsiri",
                xaxis_title="Ortalama |SHAP dəyəri|",
                height=420, margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        elif hasattr(model, "feature_importances_"):
            feats = feat_cols or list(X.columns)
            n = min(len(feats), len(model.feature_importances_))
            imp_df = (
                pd.DataFrame({"feature": feats[:n], "importance": model.feature_importances_[:n]})
                .sort_values("importance").tail(15)
            )
            fig_imp = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Blues",
                title="Xüsusiyyət Əhəmiyyəti (Gini)",
            )
            fig_imp.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Xüsusiyyət əhəmiyyətini görmək üçün SHAP aktivləşdirilmiş öyrətməni işə salın.")

    # Xəta Analizi
    st.subheader("Xəta Analizi")
    err_df = load_errors()
    if not err_df.empty:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            hourly_err = err_df.groupby("hour")["abs_error"].mean().reset_index()
            fig_err = px.bar(
                hourly_err, x="hour", y="abs_error",
                title="Günün Saatına Görə OAX",
                labels={"abs_error": "OAX (μg/m³)", "hour": "Saat"},
                color="abs_error", color_continuous_scale="OrRd",
            )
            fig_err.update_layout(height=260, coloraxis_showscale=False,
                                  margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_err, use_container_width=True)

        with col_e2:
            if "season" in err_df.columns:
                season_map = {"Winter": "Qış", "Spring": "Yaz", "Summer": "Yay", "Autumn": "Payız"}
                err_df["season_az"] = err_df["season"].map(season_map).fillna(err_df["season"])
                season_err = err_df.groupby("season_az")["abs_error"].mean().reset_index()
                fig_s = px.bar(
                    season_err, x="season_az", y="abs_error",
                    title="Mövsümə Görə OAX",
                    labels={"abs_error": "OAX (μg/m³)", "season_az": "Mövsüm"},
                    color="abs_error", color_continuous_scale="OrRd",
                )
                fig_s.update_layout(height=260, coloraxis_showscale=False,
                                    margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("Xəta analizini yaratmaq üçün öyrətməni işə salın.")

    # Model Məhdudiyyətləri
    with st.expander("Model Məhdudiyyətləri və Risk Açıqlamaları"):
        st.markdown("""
        | Risk | Təsvir | Azaldma |
        |------|--------|---------|
        | **Məlumat sızması** | Gecikmə xüsusiyyətləri diqqətlə sürüşdürülüb; rolling pəncərələr t-1-dən başlayır | `features.py`-da yoxlanılıb |
        | **Demo məlumatları** | WAQI tarixi mövcud olmadıqda PM2.5 sünidir | UI-da etiketlənib; real-vaxt oxunuşu əlavə edilib |
        | **Model sürüşməsi** | Hava keyfiyyəti nümunələri siyasət dəyişiklikləri ilə dəyişir | CI/CD vasitəsilə həftəlik yenidən öyrətmə |
        | **Korrelyasiya ≠ kauzallıq** | Statistik əlaqələr, səbəb-nəticə mexanizmləri deyil | Bütün nəticələrdə açıqlama var |
        | **Tək stansiya** | Bakı üçün bir WAQI stansiyası; məkan qatı interpolyasiya edilib | IDW aydın şəkildə təxmini kimi etiketlənib |
        | **Xəta yığılması** | Rekursiv çox addımlı proqnoz >12 saatdan sonra zəifləyir | Qeyri-müəyyənlik intervalları göstərilib |
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Biznes Mərkəzi
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    latest_pm25 = float(df["pm25"].iloc[-1])
    risk        = classify_risk(latest_pm25)
    recs        = get_recommendations(risk["risk"])

    st.subheader("Real Vaxtlı Hava Keyfiyyəti Kəşfiyyatı")

    col_b1, col_b2 = st.columns([1, 1])

    with col_b1:
        st.markdown("#### Cari Risk Vəziyyəti")
        st.markdown(f"""
        <div style="background:{risk['color']}15;border-left:5px solid {risk['color']};
                    border-radius:0 12px 12px 0;padding:1.2rem 1.4rem;margin-bottom:1rem">
          <span class="risk-badge" style="background:{risk['color']};font-size:1rem">
            {risk['label']}
          </span>
          <p style="margin:0.6rem 0 0;font-size:1.4rem;font-weight:700;color:{risk['color']}">
            {latest_pm25:.1f} μg/m³ PM2.5
          </p>
          <p style="margin:0.2rem 0 0;color:#555;font-size:0.85rem">
            {risk['who_ratio']}× ÜST illik norması
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Sağlamlıq Tövsiyələri")
        st.info(recs["health"])

        st.markdown("#### Siyasət Tövsiyələri")
        st.warning(recs["policy"])

    with col_b2:
        st.markdown("#### İnteraktiv PM2.5 Simulatoru")
        sim_val = st.slider("PM2.5 səviyyəsini simulasiya edin (μg/m³)", 0.0, 200.0,
                            float(latest_pm25), step=0.5)
        sim_risk = classify_risk(sim_val)
        sim_recs = get_recommendations(sim_risk["risk"])
        st.markdown(f"""
        <div style="background:{sim_risk['color']}15;border:2px solid {sim_risk['color']};
                    border-radius:12px;padding:1rem;text-align:center">
          <span class="risk-badge" style="background:{sim_risk['color']}">{sim_risk['label']}</span>
          <p style="margin:0.5rem 0 0;font-size:0.85rem">{sim_recs['health']}</p>
          <p style="margin:0.3rem 0 0;font-size:0.8rem;color:#888">{sim_recs['policy']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ÜST Hədd Cədvəli
    st.markdown("#### ÜST PM2.5 Tədbirləri Çərçivəsi")
    threshold_data = []
    for t in WHO_THRESHOLDS:
        threshold_data.append({
            "Risk Səviyyəsi":  t["label"],
            "PM2.5 (μg/m³)": f"{t['min']}–{t['max'] if t['max'] < 9999 else '∞'}",
            "Maraqlı Tərəf": {
                "Good":          "Ümumi ictimaiyyət",
                "Moderate":      "Məktəblər, xəstəxanalar",
                "Unhealthy":     "Bələdiyyə, səhiyyə şöbəsi",
                "Very Unhealthy":"Nazirlik, SOCAR, sənaye",
                "Hazardous":     "Milli fövqəladə hallar idarəsi",
            }.get(t["label"], ""),
            "Tövsiyə Edilən Tədbirlər": t["action"],
        })
    st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)

    st.divider()

    # ESG / Biznes göstəriciləri
    st.markdown("#### ESG və Biznes Kəşfiyyatı")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    df_30 = df.tail(30 * 24)
    exceedance_rate  = (df_30["pm25"] > 12).mean() * 100
    critical_hours   = int((df_30["pm25"] > 55).sum())
    avg_pm25_30d     = df_30["pm25"].mean()
    trend_7d         = df["pm25"].tail(168).mean() - df["pm25"].tail(336).head(168).mean()

    col_e1.metric("30 Günlük Ort. PM2.5", f"{avg_pm25_30d:.1f} μg/m³")
    col_e2.metric("ÜST Orta Hədd Aşımı", f"{exceedance_rate:.0f}%",
                  delta=f"{exceedance_rate - 50:.0f}pp etalona nəzərən", delta_color="inverse")
    col_e3.metric("Kritik Saatlar (30 gün)", f"{critical_hours} saat",
                  delta="Açıq havadan çəkinin" if critical_hours > 24 else "Təhlükəsiz hədd daxilində",
                  delta_color="inverse" if critical_hours > 24 else "normal")
    col_e4.metric("7 Günlük Trend", f"{trend_7d:+.1f} μg/m³",
                  delta="Pisləşir" if trend_7d > 2 else "Yaxşılaşır" if trend_7d < -2 else "Sabit",
                  delta_color="inverse" if trend_7d > 2 else "normal")

    # Kauzal açıqlama
    st.markdown("""
    <div style="background:#fff3cd;border-left:4px solid #ffc107;
                border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-top:1rem">
      <b>⚠️ Kauzallıq Açıqlaması:</b> Bu model PM2.5 ilə meteoroloji dəyişənlər arasında
      <em>statistik korrelyasiyaları</em> müəyyən edir. Səbəb-nəticə əlaqələri qurmur.
      Siyasət qərarları sahə ekspertləri və ölçmələri ilə təsdiqlənməlidir.
    </div>
    """, unsafe_allow_html=True)


# ── Altbilgi ──────────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.caption(f"**AirWatch AZ v2.0** · {datetime.now().strftime('%d %b %Y, %H:%M')}")
col_f2.caption("Məlumat: WAQI API + Open-Meteo · Model: korrelyasiya əsaslı, kauzal deyil")
col_f3.caption(f"Aktiv model: {model_name or 'Yoxdur'} · Xüsusiyyətlər: {len(feat_cols)}")
