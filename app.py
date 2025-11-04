import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from analytics.kpis import prepare_financials, compute_kpis
from analytics.forecast import forecast_financials, scenario_adjust

st.set_page_config(page_title="Fin Analytics & Forecast", layout="wide")

st.title("ИС: Аналитика финансовых результатов и прогноз")

with st.sidebar:
    st.header("Параметры")
    horizon = st.slider("Горизонт прогноза (мес.)", 3, 24, 12)
    seasonal_periods = st.selectbox("Сезонность (период)", [6, 12], index=1)
    trend = st.selectbox("Тренд", ["add", "mul"], index=0)
    seasonal = st.selectbox("Сезонность", ["add", "mul"], index=0)

    st.subheader("Сценарии (мес. корректировки)")
    rev_delta = st.number_input("Δ к росту выручки, % в мес.", value=0.0, step=0.5)
    opex_infl = st.number_input("Инфляция ОПЕКС, % в мес.", value=0.0, step=0.5)

    st.subheader("Загрузка данных")
    uploaded = st.file_uploader("CSV с колонками: date,revenue,cogs,opex,net_income", type=["csv"])
    use_sample = st.checkbox("Использовать демо-данные", value=(uploaded is None))

# Загрузка и подготовка данных
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
elif use_sample:
    df_raw = pd.read_csv("data/sample_pnl.csv")
else:
    st.warning("Загрузите CSV или включите демо-данные.")
    st.stop()

mapping = {
    "date": "date",
    "revenue": "revenue",
    "cogs": "cogs",
    "opex": "opex",
    "net_income": "net_income",
}
df = prepare_financials(df_raw, mapping)

if df.empty:
    st.error("После подготовки данных таблица пуста. Проверьте формат дат и названия колонок.")
    st.stop()

# KPI
kpis = compute_kpis(df)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("LTM Выручка", f"{kpis['ltm_revenue']:,.0f}")
col2.metric("YoY Выручка", f"{kpis['yoy_revenue']*100:.1f}%")
col3.metric("Валовая маржа (LTM)", f"{kpis['gross_margin_ltm']*100:.1f}%")
col4.metric("EBIT маржа (LTM)", f"{kpis['ebit_margin_ltm']*100:.1f}%")
col5.metric("Чистая маржа (LTM)", f"{kpis['net_margin_ltm']*100:.1f}%")

st.subheader("Динамика показателей")
plot_cols = ["revenue", "gross_profit", "ebit", "net_income"]
plot_df = df[plot_cols].reset_index().melt(id_vars="date", var_name="metric", value_name="value")
fig = px.line(plot_df, x="date", y="value", color="metric", title="История показателей")
fig.update_layout(legend_title_text="Метрика")
st.plotly_chart(fig, use_container_width=True)

# Прогноз
st.subheader("Прогноз (ETS)")
forecast_df, fitted_df = forecast_financials(
    df,
    horizon=horizon,
    seasonal_periods=seasonal_periods,
    trend=trend,
    seasonal=seasonal,
)

# Сценарная корректировка
forecast_scen = scenario_adjust(forecast_df, rev_monthly_pct=rev_delta, opex_monthly_pct=opex_infl)

tab1, tab2 = st.tabs(["Базовый прогноз", "Сценарий (скорректирован)"])

with tab1:
    pf = forecast_df.reset_index().melt(id_vars="date", var_name="metric", value_name="value")
    figf = px.line(pf, x="date", y="value", color="metric", title="Базовый прогноз")
    st.plotly_chart(figf, use_container_width=True)
    st.dataframe(forecast_df.tail(horizon))

with tab2:
    pfs = forecast_scen.reset_index().melt(id_vars="date", var_name="metric", value_name="value")
    figs = px.line(pfs, x="date", y="value", color="metric", title="Сценарный прогноз")
    st.plotly_chart(figs, use_container_width=True)
    st.dataframe(forecast_scen.tail(horizon))
