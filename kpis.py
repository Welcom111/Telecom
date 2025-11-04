from __future__ import annotations
import pandas as pd
import numpy as np

def _to_monthly(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    # Нормализуем к началу месяца
    df.index = df.index.to_period("M").to_timestamp()
    # Аггрегируем по месяцу (суммы)
    monthly = df.resample("MS").sum()
    return monthly

def prepare_financials(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    required = ["date", "revenue", "cogs", "opex", "net_income"]
    for k in required:
        if mapping.get(k) not in df_raw.columns:
            return pd.DataFrame()

    dfm = _to_monthly(df_raw[ [mapping["date"], mapping["revenue"], mapping["cogs"], mapping["opex"], mapping["net_income"]] ],
                      mapping["date"])
    dfm.columns = ["revenue", "cogs", "opex", "net_income"]

    # Базовые вычисления
    dfm["gross_profit"] = dfm["revenue"] - dfm["cogs"]
    dfm["ebit"] = dfm["revenue"] - dfm["cogs"] - dfm["opex"]

    # Заполнение пропусков нулями (для стабильности маржинальности)
    dfm = dfm.fillna(0.0)
    return dfm

def _safe_div(a: float, b: float) -> float:
    return np.nan if (b is None or b == 0) else a / b

def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    # LTM окна
    ltm_window = 12 if len(df) >= 12 else len(df)

    ltm_revenue = float(df["revenue"].tail(ltm_window).sum())
    ltm_gross = float(df["gross_profit"].tail(ltm_window).sum())
    ltm_ebit = float(df["ebit"].tail(ltm_window).sum())
    ltm_net = float(df["net_income"].tail(ltm_window).sum())

    gross_margin_ltm = _safe_div(ltm_gross, ltm_revenue) or 0.0
    ebit_margin_ltm = _safe_div(ltm_ebit, ltm_revenue) or 0.0
    net_margin_ltm = _safe_div(ltm_net, ltm_revenue) or 0.0

    # YoY по выручке (последний месяц vs тот же месяц год назад)
    if len(df) >= 13:
        last_rev = df["revenue"].iloc[-1]
        year_ago_rev = df["revenue"].iloc[-13]
        yoy_revenue = _safe_div(last_rev - year_ago_rev, year_ago_rev) or 0.0
    else:
        yoy_revenue = np.nan

    # CAGR по выручке (по годам)
    if df.index.min() is not None and df.index.max() is not None and len(df) >= 13:
        start = float(df["revenue"].iloc[0])
        end = float(df["revenue"].iloc[-1])
        years = max((df.index[-1].year - df.index[0].year), 1)
        cagr = (end / start) ** (1 / years) - 1 if start > 0 else np.nan
    else:
        cagr = np.nan

    return {
        "ltm_revenue": ltm_revenue,
        "gross_margin_ltm": gross_margin_ltm,
        "ebit_margin_ltm": ebit_margin_ltm,
        "net_margin_ltm": net_margin_ltm,
        "yoy_revenue": yoy_revenue,
        "cagr_revenue": cagr,
    }
