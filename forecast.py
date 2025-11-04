def _fit_ets(series: pd.Series, seasonal_periods=12, trend="add", seasonal="add"):
    series = series.astype(float)
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Минимальная длина для ETS
    if len(series.dropna()) < max(3 * seasonal_periods, 24):
        # Фолбэк: простая эксп. сглаживание без сезонности
        model = ExponentialSmoothing(series, trend=trend, seasonal=None, initialization_method="estimated")
    else:
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
    fitted = model.fit(optimized=True)
    return fitted

def forecast_financials(df: pd.DataFrame, horizon=12, seasonal_periods=12, trend="add", seasonal="add"):
    hist = df.copy()
    idx = hist.index

    result = {}
    fitted_vals = {}
    for col in ["revenue", "opex", "net_income"]:
        try:
            fitted = _fit_ets(hist[col], seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal)
            fc = fitted.forecast(horizon)
            result[col] = fc
            fitted_vals[col] = fitted.fittedvalues
        except Exception:
            # Фолбэк: наивный прогноз последнего значения
            last_v = hist[col].iloc[-1] if len(hist[col]) else 0.0
            result[col] = pd.Series([last_v] * horizon, index=pd.date_range(start=idx[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"))
            fitted_vals[col] = hist[col]

    forecast_index = pd.date_range(start=idx[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    forecast_df = pd.DataFrame(result, index=forecast_index)
    # Пересчет производных метрик
    forecast_df["cogs"] = np.nan  # опционально, если есть исторические COGS — можно прогнозировать аналогично
    # Аппроксимируем валовую прибыль как NetIncome + OPEX (если нет процентов/налогов, допустимо для демо)
    forecast_df["gross_profit"] = np.nan
    forecast_df["ebit"] = forecast_df["revenue"] - forecast_df["opex"]

    fitted_df = pd.DataFrame(fitted_vals, index=hist.index)
    return forecast_df, fitted_df

def scenario_adjust(forecast_df: pd.DataFrame, rev_monthly_pct: float = 0.0, opex_monthly_pct: float = 0.0) -> pd.DataFrame:
    scen = forecast_df.copy()
    if rev_monthly_pct != 0.0:
        factors = (1 + rev_monthly_pct / 100.0) ** (np.arange(1, len(scen) + 1))
        scen["revenue"] = scen["revenue"].values * factors
    if opex_monthly_pct != 0.0:
        factors = (1 + opex_monthly_pct / 100.0) ** (np.arange(1, len(scen) + 1))
        scen["opex"] = scen["opex"].values * factors
    scen["ebit"] = scen["revenue"] - scen["opex"]
    return scen
