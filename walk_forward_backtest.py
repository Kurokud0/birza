import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

from db import get_connection

TRADING_DAYS = 252
RISK_AVERSION = 2.5
TAU = 0.05
MAX_WEIGHT = 0.10
COVARIANCE_WINDOW = 252  # максимальное окно оценки ковариации
MIN_COVARIANCE_OBS = 180  # минимум для раннего старта с Ledoit-Wolf shrinkage
FACTOR_WINDOW = 756
MIN_REGRESSION_OBS = 180
HAC_LAGS = 5
CVAR_ALPHA = 0.95
TRANSACTION_COST = 0.001
START_BACKTEST_DATE = None  # None: начать с самой ранней допустимой даты
ARIMA_WINDOW = 500
ARIMA_ORDER = (1, 0, 1)
ARIMA_SIGNAL_SHRINK = 0.05
ARIMA_SIGNAL_CAP = 0.05
XGB_SIGNAL_SCALE = 0.05
MAX_RF_WEIGHT = 0.30  # основной сценарий
RF_CAPS = (0.20, MAX_RF_WEIGHT, 0.40)  # анализ чувствительности к ограничению RF
XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.01,
    "reg_lambda": 5.0,
}
BL_WEIGHT = 0.50
ARIMA_WEIGHT = 0.30
XGB_WEIGHT = 0.20
ML_WINDOW = 756
MIN_ML_TRAIN = 180
RANDOM_SEED = 42
RF_EPS = 1e-8
EPS = 1e-12

ML_FEATURES = [
    "return_1d", "return_3d", "return_5d", "volatility_5", "volatility_20",
    "ma_ratio_5", "ma_ratio_20"
]

SQL_PRICES = """
SELECT i.ticker, mp.trading_date, mp.close
FROM core.market_prices mp
JOIN core.instruments i ON i.instrument_id = mp.instrument_id
WHERE mp.close IS NOT NULL
ORDER BY mp.trading_date, i.ticker
"""

SQL_ML_DATA = """
SELECT ticker, trading_date, close, volume,
       usd_rate, gold_price, key_rate
FROM mart.market_daily
WHERE ticker IS NOT NULL AND trading_date IS NOT NULL
ORDER BY trading_date, ticker
"""

SQL_FACTORS = """
SELECT trading_date, mkt_rf, smb, hml, rmw, cma, rf
FROM mart.ff5_factors
WHERE trading_date IS NOT NULL
ORDER BY trading_date
"""

SQL_FUNDAMENTALS = """
SELECT ticker, report_year, period_end, market_cap
FROM core.fundamentals
WHERE ticker IS NOT NULL
ORDER BY ticker, report_year, period_end
"""

SQL_IMOEX = """
SELECT trading_date, close
FROM core.market_indices
WHERE index_code = 'IMOEX' AND close IS NOT NULL
ORDER BY trading_date
"""


def read_sql_df(sql, conn):
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [x[0] for x in cur.description]
    return pd.DataFrame(rows, columns=cols)


def to_float(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def db_float(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def db_int(value):
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def fmt_pct(value):
    return "N/A" if value is None or pd.isna(value) else f"{float(value) * 100:.2f}%"


def fmt_num(value):
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.3f}"


def prepare_prices(df):
    if df.empty:
        raise RuntimeError("core.market_prices не содержит данных.")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df["close"] = df["close"].map(to_float)
    df = df.dropna(subset=["ticker", "trading_date", "close"])
    return (
        df[df["close"] > 0]
        .sort_values(["ticker", "trading_date"])
        .drop_duplicates(["ticker", "trading_date"], keep="last")
        .reset_index(drop=True)
    )


def prepare_ml_data(df):
    if df.empty:
        raise RuntimeError("mart.market_daily не содержит данных.")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    for col in ["close", "volume", "usd_rate", "gold_price", "key_rate"]:
        df[col] = df[col].map(to_float)
    df = (
        df.dropna(subset=["ticker", "trading_date", "close"])
        .sort_values(["ticker", "trading_date"])
        .drop_duplicates(["ticker", "trading_date"], keep="last")
        .reset_index(drop=True)
    )
    grouped = df.groupby("ticker", sort=False)["close"]
    df["return_1d"] = grouped.pct_change(1)
    df["return_3d"] = grouped.pct_change(3)
    df["return_5d"] = grouped.pct_change(5)
    df["volatility_5"] = df.groupby("ticker", sort=False)["return_1d"].transform(lambda x: x.rolling(5).std())
    df["volatility_20"] = df.groupby("ticker", sort=False)["return_1d"].transform(lambda x: x.rolling(20).std())
    ma5 = grouped.transform(lambda x: x.rolling(5).mean())
    ma20 = grouped.transform(lambda x: x.rolling(20).mean())
    df["ma_ratio_5"] = df["close"] / ma5 - 1.0
    df["ma_ratio_20"] = df["close"] / ma20 - 1.0
    df["volume_ratio_5"] = df["volume"] / df.groupby("ticker", sort=False)["volume"].transform(lambda x: x.rolling(5).mean())
    df["target_return"] = df.groupby("ticker", sort=False)["close"].shift(-1) / df["close"] - 1.0
    df["target"] = np.nan
    labelled = df["target_return"].notna()
    df.loc[labelled, "target"] = (df.loc[labelled, "target_return"] > 0).astype(int)
    return df




def prepare_factors(df):
    if df.empty:
        raise RuntimeError("mart.ff5_factors не содержит данных.")
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        df[col] = df[col].map(to_float)
    return df.dropna(subset=["trading_date"]).sort_values("trading_date").drop_duplicates(
        "trading_date", keep="last"
    ).reset_index(drop=True)


def prepare_fundamentals(df):
    if df.empty:
        raise RuntimeError("core.fundamentals не содержит данных.")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["report_year"] = pd.to_numeric(df["report_year"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["market_cap"] = df["market_cap"].map(to_float)
    df = df.dropna(subset=["ticker", "report_year"])
    df["report_year"] = df["report_year"].astype(int)
    df["effective_date"] = pd.to_datetime(df["report_year"].map(lambda x: f"{x + 1}-07-01"))
    return df


def prepare_imoex(df):
    if df.empty:
        raise RuntimeError("IMOEX не содержит данных.")
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df["close"] = df["close"].map(to_float)
    df = df.dropna(subset=["trading_date", "close"])
    return df[df["close"] > 0].sort_values("trading_date").drop_duplicates(
        "trading_date", keep="last"
    ).reset_index(drop=True)


def build_price_pivot(prices, tickers):
    return prices.pivot_table(
        index="trading_date", columns="ticker", values="close", aggfunc="last"
    ).reindex(columns=tickers)


def build_returns(price_pivot):
    return price_pivot.pct_change().reset_index()


def get_market_caps_at_date(fundamentals, tickers, as_of_date):
    available = fundamentals[fundamentals["effective_date"] <= pd.Timestamp(as_of_date)].copy()
    if available.empty:
        raise RuntimeError(f"Нет fundamentals на {as_of_date.date()}.")
    available = (
        available.sort_values(
            ["ticker", "effective_date", "report_year", "period_end"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("ticker", keep="first")
    )
    positive = available[available["market_cap"].notna() & (available["market_cap"] > 0)]
    if positive.empty:
        raise RuntimeError(f"Нет положительных market_cap на {as_of_date.date()}.")
    median_cap = float(positive["market_cap"].median())
    rows = []
    for ticker in tickers:
        match = available[available["ticker"] == ticker]
        cap = to_float(match.iloc[0]["market_cap"]) if not match.empty else np.nan
        rows.append({
            "ticker": ticker,
            "market_cap": float(cap) if pd.notna(cap) and cap > 0 else median_cap,
        })
    result = pd.DataFrame(rows)
    result["market_weight"] = result["market_cap"] / result["market_cap"].sum()
    return result


FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma"]


def prepare_factor_history(factors, as_of_date):
    history = factors[factors["trading_date"] <= pd.Timestamp(as_of_date)].copy()
    history = history.dropna(subset=FACTOR_COLUMNS + ["rf"]).tail(FACTOR_WINDOW)
    if len(history) < MIN_REGRESSION_OBS:
        raise RuntimeError(f"На {as_of_date.date()} только {len(history)} FF5-наблюдений.")
    return history.reset_index(drop=True)


def calculate_factor_premiums(history):
    return history[FACTOR_COLUMNS].mean() * TRADING_DAYS


def calculate_current_rf(factors, as_of_date):
    history = factors[(factors["trading_date"] <= pd.Timestamp(as_of_date)) & factors["rf"].notna()]
    if history.empty:
        raise RuntimeError(f"Нет RF на {as_of_date.date()}.")
    daily_rf = float(history.iloc[-1]["rf"])
    if daily_rf <= -1:
        raise RuntimeError(f"Некорректный RF: {daily_rf}")
    return (1.0 + daily_rf) ** TRADING_DAYS - 1.0


def run_ff5_regressions(returns, factor_history, tickers):
    results = []
    for ticker in tickers:
        stock = returns[["trading_date", ticker]].rename(columns={ticker: "return"})
        merged = factor_history.merge(stock, on="trading_date", how="inner").dropna(
            subset=["return", "rf"] + FACTOR_COLUMNS
        )
        if len(merged) < MIN_REGRESSION_OBS:
            continue
        y = merged["return"] - merged["rf"]
        x = sm.add_constant(merged[FACTOR_COLUMNS].astype(float), has_constant="add")
        try:
            fit = sm.OLS(y.astype(float), x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        except Exception:
            continue
        results.append({
            "ticker": ticker,
            "alpha_annual": float(fit.params["const"] * TRADING_DAYS),
            "alpha_pvalue": float(fit.pvalues["const"]),
            "beta_mkt": float(fit.params["mkt_rf"]),
            "beta_smb": float(fit.params["smb"]),
            "beta_hml": float(fit.params["hml"]),
            "beta_rmw": float(fit.params["rmw"]),
            "beta_cma": float(fit.params["cma"]),
            "r_squared": float(fit.rsquared),
            "n_obs": int(len(merged)),
        })
    if not results:
        raise RuntimeError("Ни одна акция не прошла FF5-регрессию.")
    return pd.DataFrame(results)


def build_ff5_expected_returns(regression, premiums, rf_annual):
    rows = []
    for _, row in regression.iterrows():
        alpha = row["alpha_annual"] if row["alpha_pvalue"] < 0.05 else 0.0
        excess = (
            alpha + row["beta_mkt"] * premiums["mkt_rf"]
            + row["beta_smb"] * premiums["smb"]
            + row["beta_hml"] * premiums["hml"]
            + row["beta_rmw"] * premiums["rmw"]
            + row["beta_cma"] * premiums["cma"]
        )
        rows.append({"ticker": row["ticker"], "ff5_view_return": float(rf_annual + excess)})
    return pd.DataFrame(rows)


def build_covariance_at_date(returns, tickers, as_of_date):
    history = returns[returns["trading_date"] <= pd.Timestamp(as_of_date)].dropna(subset=tickers)
    if len(history) < MIN_COVARIANCE_OBS:
        raise RuntimeError(
            f"На {as_of_date.date()} только {len(history)} общих наблюдений; "
            f"необходимо минимум {MIN_COVARIANCE_OBS}."
        )
    history = history.tail(COVARIANCE_WINDOW)
    estimator = LedoitWolf(assume_centered=False).fit(history[tickers].astype(float).values)
    covariance = pd.DataFrame(
        estimator.covariance_ * TRADING_DAYS,
        index=tickers,
        columns=tickers,
    )
    return covariance, history


def calculate_black_litterman(covariance, market_weights, ff5_expected, rf_annual):
    tickers = ff5_expected["ticker"].tolist()
    sigma = covariance.loc[tickers, tickers].values.astype(float)
    weights = market_weights.set_index("ticker").loc[tickers, "market_weight"].astype(float).values
    views = ff5_expected.set_index("ticker").loc[tickers, "ff5_view_return"].astype(float).values
    pi_excess = RISK_AVERSION * sigma.dot(weights)
    q_excess = views - rf_annual
    tau_sigma = TAU * sigma
    omega = np.maximum(TAU * np.diag(sigma), 1e-12)
    prior_precision = np.linalg.solve(tau_sigma, np.eye(len(tickers)))
    omega_inv = np.diag(1.0 / omega)
    posterior_precision = prior_precision + omega_inv
    rhs = prior_precision.dot(pi_excess) + omega_inv.dot(q_excess)
    posterior_excess = np.linalg.solve(posterior_precision, rhs)
    return pd.DataFrame({
        "ticker": tickers,
        "market_weight": weights,
        "ff5_view_return": views,
        "equilibrium_return": rf_annual + pi_excess,
        "posterior_return": rf_annual + posterior_excess,
    })


def fit_arima(history):
    history = np.asarray(history, dtype=float)
    history = history[np.isfinite(history)][-ARIMA_WINDOW:]
    if len(history) < 30:
        return None
    for order in [ARIMA_ORDER, (1, 0, 0)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ARIMA(
                    history,
                    order=order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(method_kwargs={"maxiter": 200})
        except Exception:
            pass
    return None


def predict_arima(ml_data, ticker, as_of_date):
    stock = ml_data[
        (ml_data["ticker"] == ticker)
        & (ml_data["trading_date"] <= pd.Timestamp(as_of_date))
    ].sort_values("trading_date")
    history = stock["close"].pct_change().dropna().tail(ARIMA_WINDOW).to_numpy(dtype=float)
    model = fit_arima(history)
    if model is None:
        prediction = float(history[-1]) if len(history) else 0.0
    else:
        try:
            prediction = float(model.forecast(steps=1)[0])
        except Exception:
            prediction = float(history[-1]) if len(history) else 0.0
    if not np.isfinite(prediction):
        prediction = float(history[-1]) if len(history) else 0.0
    raw_linear_annual = float(prediction * TRADING_DAYS)
    alpha_annual = float(np.clip(raw_linear_annual * ARIMA_SIGNAL_SHRINK, -ARIMA_SIGNAL_CAP, ARIMA_SIGNAL_CAP))
    return prediction, raw_linear_annual, alpha_annual


def predict_xgb(ml_data, ticker, as_of_date, rf_annual):
    stock = ml_data[ml_data["ticker"] == ticker].sort_values("trading_date").copy()
    train = stock[stock["trading_date"] < pd.Timestamp(as_of_date)].dropna(subset=ML_FEATURES + ["target"])
    current = stock[stock["trading_date"] <= pd.Timestamp(as_of_date)].dropna(subset=ML_FEATURES)
    if len(train) < MIN_ML_TRAIN:
        raise RuntimeError(f"Для {ticker} только {len(train)} обучающих строк XGBoost.")
    if current.empty:
        raise RuntimeError(f"Нет признаков XGBoost для {ticker} на {as_of_date.date()}.")
    train = train.tail(ML_WINDOW)
    model = XGBClassifier(
        **XGB_PARAMS,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(train[ML_FEATURES].astype(float), train["target"].astype(int))
    row = current.iloc[[-1]]
    probability = model.predict_proba(row[ML_FEATURES].astype(float))[0]
    classes = list(model.classes_)
    p_up = float(probability[classes.index(1)]) if 1 in classes else 0.0
    p_up = float(np.clip(p_up, 0.0, 1.0))
    alpha_annual = XGB_SIGNAL_SCALE * (2.0 * p_up - 1.0)
    return p_up, float(alpha_annual), float(rf_annual + alpha_annual), int(len(train)), pd.Timestamp(row["trading_date"].iloc[0]).date()


def optimize_portfolio(expected_returns, covariance, rf_annual, max_rf_weight):
    n = len(expected_returns)
    minimum_stock_weight = 1.0 - max_rf_weight
    initial = np.repeat(1.0 / n, n)

    def utility(weights):
        expected_total = rf_annual + weights @ (expected_returns - rf_annual)
        variance = max(float(weights @ covariance.to_numpy() @ weights), 0.0)
        return -(expected_total - RISK_AVERSION / 2.0 * variance)

    result = minimize(
        utility,
        initial.astype(float),
        method="SLSQP",
        bounds=[(0.0, MAX_WEIGHT)] * n,
        constraints=[
            {"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)},
            {"type": "ineq", "fun": lambda w: np.sum(w) - minimum_stock_weight},
        ],
        options={"maxiter": 2000, "ftol": 1e-10, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Оптимизация не сошлась: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, MAX_WEIGHT)
    stock_sum = float(weights.sum())
    if stock_sum > 1.0 + 1e-8:
        raise RuntimeError("Сумма весов акций превышает 100%.")
    if stock_sum < minimum_stock_weight - 1e-6:
        raise RuntimeError("Не выполнено ограничение минимальной доли акций.")
    return weights, max(0.0, 1.0 - stock_sum), result


def calculate_rebalance_portfolios(rebalance_date, returns, factors, fundamentals, ml_data, tickers):
    factor_history = prepare_factor_history(factors, rebalance_date)
    premiums = calculate_factor_premiums(factor_history)
    rf_annual = calculate_current_rf(factors, rebalance_date)
    regression = run_ff5_regressions(returns, factor_history, tickers)
    missing = sorted(set(tickers) - set(regression["ticker"]))
    if missing:
        raise RuntimeError("Не удалось построить FF5 для: " + ", ".join(missing))
    regression = regression.set_index("ticker").loc[tickers].reset_index()
    ff5_expected = build_ff5_expected_returns(regression, premiums, rf_annual)
    market_weights = get_market_caps_at_date(fundamentals, tickers, rebalance_date)
    covariance, covariance_history = build_covariance_at_date(returns, tickers, rebalance_date)
    bl = calculate_black_litterman(covariance, market_weights, ff5_expected, rf_annual)

    signal_rows = []
    for ticker in tickers:
        arima_daily, arima_raw_annual, arima_alpha = predict_arima(ml_data, ticker, rebalance_date)
        p_up, xgb_alpha, xgb_expected, train_rows, feature_date = predict_xgb(
            ml_data, ticker, rebalance_date, rf_annual
        )
        signal_rows.append({
            "ticker": ticker,
            "arima_pred_daily": arima_daily,
            "arima_pred_annual_raw": arima_raw_annual,
            "arima_alpha_annual": arima_alpha,
            "arima_expected_return_annual": rf_annual + arima_alpha,
            "xgb_probability_growth": p_up,
            "xgb_probability_down": 1.0 - p_up,
            "xgb_signal_alpha_annual": xgb_alpha,
            "xgb_expected_return_annual": xgb_expected,
            "ml_train_rows": train_rows,
            "ml_feature_date": feature_date,
        })
    signals = pd.DataFrame(signal_rows)
    base = bl.merge(signals, on="ticker", how="inner", validate="one_to_one")
    base = base.merge(
        regression[["ticker", "alpha_annual", "alpha_pvalue", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "r_squared", "n_obs"]],
        on="ticker", how="left", validate="one_to_one"
    )
    base["rf_annual"] = rf_annual
    base["factor_premium_mkt"] = premiums["mkt_rf"]
    base["factor_premium_smb"] = premiums["smb"]
    base["factor_premium_hml"] = premiums["hml"]
    base["factor_premium_rmw"] = premiums["rmw"]
    base["factor_premium_cma"] = premiums["cma"]
    base["factor_observations"] = len(factor_history)
    base["covariance_observations"] = len(covariance_history)
    base["ml_feature_count"] = len(ML_FEATURES)
    base["rebalance_date"] = pd.Timestamp(rebalance_date).date()

    n = len(tickers)
    configs = {}
    signal_sets = [
        ("BL", "BL_FREE_RF", 1.00, 0.00, 0.00),
        ("BL_ARIMA", "BL_ARIMA_FREE_RF", 0.70, 0.30, 0.00),
        ("BL_ARIMA_XGB", "BL_ARIMA_XGB_FREE_RF", 0.50, 0.30, 0.20),
    ]
    for prefix, free_name, w_bl, w_arima, w_xgb in signal_sets:
        configs[free_name] = (w_bl, w_arima, w_xgb, 1.00)
        for rf_cap in RF_CAPS:
            configs[f"{prefix}_RF{int(round(rf_cap * 100))}"] = (
                w_bl, w_arima, w_xgb, rf_cap
            )
    results = {}
    bl_values = base["posterior_return"].astype(float).to_numpy()
    arima_values = base["arima_expected_return_annual"].astype(float).to_numpy()
    xgb_values = base["xgb_expected_return_annual"].astype(float).to_numpy()

    for name, (w_bl, w_arima, w_xgb, max_rf) in configs.items():
        expected_returns = w_bl * bl_values + w_arima * arima_values + w_xgb * xgb_values
        weights, rf_weight, _ = optimize_portfolio(expected_returns, covariance, rf_annual, max_rf)
        result = base.copy()
        result["portfolio_name"] = name
        result["max_rf_weight"] = max_rf
        result["combined_expected_return"] = expected_returns
        result["weight"] = weights
        result["risk_free_weight"] = rf_weight
        result["stock_weight_sum"] = float(weights.sum())
        result["portfolio_expected_return"] = float(rf_annual + weights @ (expected_returns - rf_annual))
        result["expected_return_contribution"] = result["weight"] * result["combined_expected_return"]
        results[name] = result

    # Нерегулируемые контрольные варианты для той же даты и того же универсума.
    base_expected = base.copy()
    base_expected["combined_expected_return"] = bl_values
    for name, weights, rf_weight, max_rf in [
        ("EQUAL_WEIGHT_STOCKS", np.repeat(1.0 / n, n), 0.0, 0.0),
        ("RF_ONLY", np.zeros(n), 1.0, 1.0),
    ]:
        result = base_expected.copy()
        result["portfolio_name"] = name
        result["max_rf_weight"] = max_rf
        result["weight"] = weights
        result["risk_free_weight"] = rf_weight
        result["stock_weight_sum"] = float(weights.sum())
        result["portfolio_expected_return"] = float(rf_annual + weights @ (bl_values - rf_annual))
        result["expected_return_contribution"] = result["weight"] * result["combined_expected_return"]
        results[name] = result
    return results




def calculate_turnover(previous_weights, current_weights, tickers, previous_rf, current_rf):
    previous = pd.Series(0.0, index=tickers) if previous_weights is None else previous_weights.reindex(tickers, fill_value=0.0).astype(float)
    current = current_weights.reindex(tickers, fill_value=0.0).astype(float)
    return float(np.clip(
        0.5 * (np.abs(current.values - previous.values).sum() + abs(float(current_rf) - float(previous_rf))),
        0.0, 1.0
    ))


def calculate_backtest_daily(returns, rebalance_results, rebalance_dates, factors, imoex):
    rows = []
    previous_weights = None
    previous_rf = 1.0
    benchmark = imoex[["trading_date", "close"]].copy().sort_values("trading_date")
    benchmark["imoex_return"] = benchmark["close"].pct_change()
    rf_daily = factors[["trading_date", "rf"]].dropna().drop_duplicates("trading_date", keep="last")
    rf_daily = rf_daily.sort_values("trading_date")

    for i, result in enumerate(rebalance_results):
        if i + 1 >= len(rebalance_dates):
            break
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]
        weights = result.set_index("ticker")["weight"].astype(float)
        current_rf = float(result["risk_free_weight"].iloc[0])
        tickers = list(weights.index)
        turnover = calculate_turnover(previous_weights, weights, tickers, previous_rf, current_rf)
        transaction_cost = turnover * TRANSACTION_COST

        period = returns[
            (returns["trading_date"] > current_date)
            & (returns["trading_date"] <= next_date)
        ].copy().dropna(subset=tickers)
        if period.empty:
            previous_weights = weights
            previous_rf = current_rf
            continue

        daily = period[["trading_date"]].copy()
        daily["stock_return"] = period[tickers].mul(weights, axis=1).sum(axis=1).to_numpy()
        daily = pd.merge_asof(
            daily.sort_values("trading_date"), rf_daily,
            on="trading_date", direction="backward"
        ).reset_index(drop=True)
        daily["rf_weight"] = current_rf
        daily["portfolio_gross_return"] = daily["stock_return"] + current_rf * daily["rf"].fillna(0.0)
        daily["turnover"] = 0.0
        daily["transaction_cost"] = 0.0
        if len(daily):
            first_idx = daily.index[0]
            daily.loc[first_idx, "turnover"] = turnover
            daily.loc[first_idx, "transaction_cost"] = transaction_cost
        daily["portfolio_net_return"] = daily["portfolio_gross_return"] - daily["transaction_cost"]
        daily["rebalance_date"] = current_date.date()
        daily["next_rebalance_date"] = next_date.date()
        daily["transaction_cost_rate"] = TRANSACTION_COST
        daily = daily.merge(benchmark[["trading_date", "imoex_return"]], on="trading_date", how="left")
        daily["portfolio_name"] = str(result["portfolio_name"].iloc[0])
        rows.append(daily)
        previous_weights = weights
        previous_rf = current_rf

    if not rows:
        raise RuntimeError("Не удалось сформировать daily backtest.")
    daily = (
        pd.concat(rows, ignore_index=True)
        .sort_values("trading_date")
        .drop_duplicates("trading_date", keep="last")
        .reset_index(drop=True)
    )
    daily["portfolio_cumulative"] = (1.0 + daily["portfolio_net_return"]).cumprod()
    daily["imoex_cumulative"] = (1.0 + daily["imoex_return"].fillna(0.0)).cumprod()
    return daily




def annualized_return(returns):
    values = pd.Series(returns).dropna().astype(float)
    if values.empty:
        return np.nan
    growth = float((1.0 + values).prod())
    years = len(values) / TRADING_DAYS
    return float(growth ** (1.0 / years) - 1.0) if growth > 0 and years > 0 else np.nan


def max_drawdown(returns):
    curve = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def var_cvar(returns, alpha=CVAR_ALPHA):
    values = pd.Series(returns).dropna().astype(float).values
    if len(values) == 0:
        return np.nan, np.nan
    losses = -values
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else var
    scale = math.sqrt(TRADING_DAYS)
    return var * scale, cvar * scale


def beta(portfolio, benchmark):
    df = pd.concat([pd.Series(portfolio, name="p"), pd.Series(benchmark, name="b")], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    var = df["b"].var(ddof=1)
    if pd.isna(var) or var <= 0:
        return np.nan
    return float(np.cov(df["p"], df["b"], ddof=1)[0, 1] / var)


def tracking_error(portfolio, benchmark):
    df = pd.concat([pd.Series(portfolio, name="p"), pd.Series(benchmark, name="b")], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    return float((df["p"] - df["b"]).std(ddof=1) * math.sqrt(TRADING_DAYS))


def calculate_backtest_metrics(daily):
    portfolio = daily["portfolio_net_return"].astype(float)
    benchmark_returns = daily["imoex_return"].astype(float)
    total_return = float((1.0 + portfolio).prod() - 1.0)
    annual_return = annualized_return(portfolio)
    benchmark_total = float((1.0 + benchmark_returns.fillna(0.0)).prod() - 1.0)
    benchmark_annual = annualized_return(benchmark_returns.dropna())
    volatility = float(portfolio.std(ddof=1) * math.sqrt(TRADING_DAYS))
    rf_daily = daily["rf"].astype(float)
    aligned = pd.concat(
        [portfolio.rename("portfolio"), rf_daily.rename("rf")], axis=1
    ).dropna()
    if len(aligned) > 1:
        excess_returns = aligned["portfolio"] - aligned["rf"]
        portfolio_daily_vol = aligned["portfolio"].std(ddof=1)
        sharpe = (
            float(excess_returns.mean() / portfolio_daily_vol * math.sqrt(TRADING_DAYS))
            if portfolio_daily_vol > EPS else np.nan
        )
    else:
        sharpe = np.nan
    rf_daily_valid = rf_daily.dropna()
    rf_annual = (
        float((1.0 + rf_daily_valid.mean()) ** TRADING_DAYS - 1.0)
        if not rf_daily_valid.empty else np.nan
    )
    var_95, cvar_95 = var_cvar(portfolio)
    rebalance_rf = daily.groupby("rebalance_date")["rf_weight"].first().astype(float)
    rebalances = int(daily["rebalance_date"].nunique())
    total_turnover = float(daily["turnover"].sum())
    return {
        "observations": int(len(daily)),
        "rebalances": rebalances,
        "total_return": total_return,
        "annual_return": annual_return,
        "benchmark_total_return": benchmark_total,
        "benchmark_annual_return": benchmark_annual,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(portfolio),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta_imoex": beta(portfolio, benchmark_returns),
        "tracking_error": tracking_error(portfolio, benchmark_returns),
        "total_turnover": total_turnover,
        "average_turnover": total_turnover / rebalances if rebalances else np.nan,
        "max_turnover": float(daily["turnover"].max()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        "rf_annual": rf_annual,
        "avg_rf_weight": float(rebalance_rf.mean()),
        "max_rf_weight": float(rebalance_rf.max()),
        "min_rf_weight": float(rebalance_rf.min()),
        "rebalances_with_rf": int((rebalance_rf > RF_EPS).sum()),
        "rebalances_rf_50": int((rebalance_rf >= 0.50).sum()),
    }


CREATE_WEIGHTS_SQL = """
DROP TABLE IF EXISTS mart.backtest_weights;
CREATE TABLE mart.backtest_weights (
    portfolio_name VARCHAR(60) NOT NULL,
    rebalance_date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    risk_free_weight DOUBLE PRECISION NOT NULL,
    max_rf_weight DOUBLE PRECISION NOT NULL,
    stock_weight_sum DOUBLE PRECISION NOT NULL,
    market_weight DOUBLE PRECISION NOT NULL,
    ff5_view_return DOUBLE PRECISION NOT NULL,
    equilibrium_return DOUBLE PRECISION NOT NULL,
    posterior_return DOUBLE PRECISION NOT NULL,
    arima_pred_daily DOUBLE PRECISION NOT NULL,
    arima_pred_annual_raw DOUBLE PRECISION NOT NULL,
    arima_alpha_annual DOUBLE PRECISION NOT NULL,
    arima_expected_return_annual DOUBLE PRECISION NOT NULL,
    xgb_probability_growth DOUBLE PRECISION NOT NULL,
    xgb_probability_down DOUBLE PRECISION NOT NULL,
    xgb_signal_alpha_annual DOUBLE PRECISION NOT NULL,
    xgb_expected_return_annual DOUBLE PRECISION NOT NULL,
    combined_expected_return DOUBLE PRECISION NOT NULL,
    expected_return_contribution DOUBLE PRECISION NOT NULL,
    portfolio_expected_return DOUBLE PRECISION NOT NULL,
    alpha_annual DOUBLE PRECISION,
    alpha_pvalue DOUBLE PRECISION,
    beta_mkt DOUBLE PRECISION,
    beta_smb DOUBLE PRECISION,
    beta_hml DOUBLE PRECISION,
    beta_rmw DOUBLE PRECISION,
    beta_cma DOUBLE PRECISION,
    r_squared DOUBLE PRECISION,
    n_obs INTEGER,
    rf_annual DOUBLE PRECISION NOT NULL,
    ml_train_rows INTEGER NOT NULL,
    ml_feature_count INTEGER NOT NULL,
    ml_feature_date DATE NOT NULL,
    factor_premium_mkt DOUBLE PRECISION NOT NULL,
    factor_premium_smb DOUBLE PRECISION NOT NULL,
    factor_premium_hml DOUBLE PRECISION NOT NULL,
    factor_premium_rmw DOUBLE PRECISION NOT NULL,
    factor_premium_cma DOUBLE PRECISION NOT NULL,
    factor_observations INTEGER NOT NULL,
    covariance_observations INTEGER NOT NULL,
    PRIMARY KEY (portfolio_name, rebalance_date, ticker)
);
"""

CREATE_DAILY_SQL = """
DROP TABLE IF EXISTS mart.backtest_daily;
CREATE TABLE mart.backtest_daily (
    portfolio_name VARCHAR(60) NOT NULL,
    trading_date DATE NOT NULL,
    rebalance_date DATE NOT NULL,
    next_rebalance_date DATE NOT NULL,
    stock_return DOUBLE PRECISION NOT NULL,
    rf DOUBLE PRECISION,
    rf_weight DOUBLE PRECISION NOT NULL,
    portfolio_gross_return DOUBLE PRECISION NOT NULL,
    transaction_cost DOUBLE PRECISION NOT NULL,
    portfolio_net_return DOUBLE PRECISION NOT NULL,
    imoex_return DOUBLE PRECISION,
    portfolio_cumulative DOUBLE PRECISION NOT NULL,
    imoex_cumulative DOUBLE PRECISION NOT NULL,
    turnover DOUBLE PRECISION NOT NULL,
    transaction_cost_rate DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (portfolio_name, trading_date)
);
"""

CREATE_METRICS_SQL = """
DROP TABLE IF EXISTS mart.backtest_metrics;
CREATE TABLE mart.backtest_metrics (
    portfolio_name VARCHAR(60) PRIMARY KEY,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    observations INTEGER NOT NULL,
    rebalances INTEGER NOT NULL,
    total_return DOUBLE PRECISION NOT NULL,
    annual_return DOUBLE PRECISION,
    benchmark_total_return DOUBLE PRECISION,
    benchmark_annual_return DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    var_95_annual_approx DOUBLE PRECISION,
    cvar_95_annual_approx DOUBLE PRECISION,
    beta_imoex DOUBLE PRECISION,
    tracking_error_annual DOUBLE PRECISION,
    total_turnover DOUBLE PRECISION,
    average_turnover DOUBLE PRECISION,
    max_turnover DOUBLE PRECISION,
    total_transaction_cost DOUBLE PRECISION,
    risk_free_rate DOUBLE PRECISION,
    avg_rf_weight DOUBLE PRECISION,
    max_rf_weight DOUBLE PRECISION,
    min_rf_weight DOUBLE PRECISION,
    rebalances_with_rf INTEGER,
    rebalances_rf_50 INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def save_backtest(weight_results, daily_results, metrics_by_strategy):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_WEIGHTS_SQL)
            cur.execute(CREATE_DAILY_SQL)
            cur.execute(CREATE_METRICS_SQL)

            weight_cols = [
                "portfolio_name", "rebalance_date", "ticker", "weight", "risk_free_weight", "max_rf_weight",
                "stock_weight_sum", "market_weight", "ff5_view_return", "equilibrium_return", "posterior_return",
                "arima_pred_daily", "arima_pred_annual_raw", "arima_alpha_annual", "arima_expected_return_annual",
                "xgb_probability_growth", "xgb_probability_down", "xgb_signal_alpha_annual", "xgb_expected_return_annual",
                "combined_expected_return", "expected_return_contribution", "portfolio_expected_return", "alpha_annual",
                "alpha_pvalue", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "r_squared", "n_obs",
                "rf_annual", "ml_train_rows", "ml_feature_count", "ml_feature_date", "factor_premium_mkt",
                "factor_premium_smb", "factor_premium_hml", "factor_premium_rmw", "factor_premium_cma",
                "factor_observations", "covariance_observations",
            ]
            int_cols = {"n_obs", "ml_train_rows", "ml_feature_count", "factor_observations", "covariance_observations"}
            date_cols = {"rebalance_date", "ml_feature_date"}
            weight_rows = []
            for _, row in weight_results.iterrows():
                values = []
                for col in weight_cols:
                    if col in {"portfolio_name", "ticker"}:
                        values.append(row[col])
                    elif col in date_cols:
                        values.append(pd.to_datetime(row[col]).date())
                    elif col in int_cols:
                        values.append(db_int(row[col]))
                    else:
                        values.append(db_float(row[col]))
                weight_rows.append(tuple(values))
            cur.executemany(
                f"INSERT INTO mart.backtest_weights ({','.join(weight_cols)}) VALUES ({','.join(['%s'] * len(weight_cols))})",
                weight_rows,
            )

            daily_cols = [
                "portfolio_name", "trading_date", "rebalance_date", "next_rebalance_date", "stock_return", "rf",
                "rf_weight", "portfolio_gross_return", "transaction_cost", "portfolio_net_return", "imoex_return",
                "portfolio_cumulative", "imoex_cumulative", "turnover", "transaction_cost_rate",
            ]
            daily_rows = []
            for _, row in daily_results.iterrows():
                values = []
                for col in daily_cols:
                    if col == "portfolio_name":
                        values.append(row[col])
                    elif col in {"trading_date", "rebalance_date", "next_rebalance_date"}:
                        values.append(pd.to_datetime(row[col]).date())
                    else:
                        values.append(db_float(row[col]))
                daily_rows.append(tuple(values))
            cur.executemany(
                f"INSERT INTO mart.backtest_daily ({','.join(daily_cols)}) VALUES ({','.join(['%s'] * len(daily_cols))})",
                daily_rows,
            )

            metric_cols = [
                "portfolio_name", "start_date", "end_date", "observations", "rebalances", "total_return", "annual_return",
                "benchmark_total_return", "benchmark_annual_return", "volatility", "sharpe_ratio", "max_drawdown",
                "var_95_annual_approx", "cvar_95_annual_approx", "beta_imoex", "tracking_error_annual", "total_turnover",
                "average_turnover", "max_turnover", "total_transaction_cost", "risk_free_rate", "avg_rf_weight",
                "max_rf_weight", "min_rf_weight", "rebalances_with_rf", "rebalances_rf_50",
            ]
            metric_rows = []
            for name, metrics in metrics_by_strategy.items():
                strategy_daily = daily_results[daily_results["portfolio_name"] == name]
                values = {
                    "portfolio_name": name,
                    "start_date": strategy_daily["trading_date"].min().date(),
                    "end_date": strategy_daily["trading_date"].max().date(),
                    "observations": metrics["observations"],
                    "rebalances": metrics["rebalances"],
                    "total_return": metrics["total_return"],
                    "annual_return": metrics["annual_return"],
                    "benchmark_total_return": metrics["benchmark_total_return"],
                    "benchmark_annual_return": metrics["benchmark_annual_return"],
                    "volatility": metrics["volatility"],
                    "sharpe_ratio": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "var_95_annual_approx": metrics["var_95"],
                    "cvar_95_annual_approx": metrics["cvar_95"],
                    "beta_imoex": metrics["beta_imoex"],
                    "tracking_error_annual": metrics["tracking_error"],
                    "total_turnover": metrics["total_turnover"],
                    "average_turnover": metrics["average_turnover"],
                    "max_turnover": metrics["max_turnover"],
                    "total_transaction_cost": metrics["total_transaction_cost"],
                    "risk_free_rate": metrics["rf_annual"],
                    "avg_rf_weight": metrics["avg_rf_weight"],
                    "max_rf_weight": metrics["max_rf_weight"],
                    "min_rf_weight": metrics["min_rf_weight"],
                    "rebalances_with_rf": metrics["rebalances_with_rf"],
                    "rebalances_rf_50": metrics["rebalances_rf_50"],
                }
                metric_rows.append(tuple(
                    values[col] if col == "portfolio_name" else values[col]
                    if col in {"start_date", "end_date"}
                    else db_int(values[col]) if col in {"observations", "rebalances", "rebalances_with_rf", "rebalances_rf_50"}
                    else db_float(values[col])
                    for col in metric_cols
                ))
            cur.executemany(
                f"INSERT INTO mart.backtest_metrics ({','.join(metric_cols)}) VALUES ({','.join(['%s'] * len(metric_cols))})",
                metric_rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()




def build_rebalance_dates(factors, imoex, start_date):
    factor_dates = pd.DatetimeIndex(pd.to_datetime(factors["trading_date"]).drop_duplicates())
    imoex_dates = pd.DatetimeIndex(pd.to_datetime(imoex["trading_date"]).drop_duplicates())
    dates = factor_dates.intersection(imoex_dates)
    dates = dates[dates >= pd.Timestamp(start_date)]
    if len(dates) == 0:
        return []
    series = pd.Series(dates)
    monthly = series.groupby(series.dt.to_period("M")).max()
    return [pd.Timestamp(x) for x in monthly.tolist()]


def main():
    print("WALK-FORWARD: СРАВНЕНИЕ СТРАТЕГИЙ\n")
    conn = get_connection()
    try:
        prices = read_sql_df(SQL_PRICES, conn)
        ml_raw = read_sql_df(SQL_ML_DATA, conn)
        factors = read_sql_df(SQL_FACTORS, conn)
        fundamentals = read_sql_df(SQL_FUNDAMENTALS, conn)
        imoex = read_sql_df(SQL_IMOEX, conn)
    finally:
        conn.close()

    prices = prepare_prices(prices)
    ml_data = prepare_ml_data(ml_raw)
    factors = prepare_factors(factors)
    fundamentals = prepare_fundamentals(fundamentals)
    imoex = prepare_imoex(imoex)

    tickers = sorted(prices["ticker"].unique().tolist())
    if len(tickers) != 25:
        raise RuntimeError(f"Ожидалось 25 акций, получено {len(tickers)}.")
    price_pivot = build_price_pivot(prices, tickers)
    returns = build_returns(price_pivot)
    complete_returns = returns.dropna(subset=tickers).copy()
    if len(complete_returns) < 500:
        raise RuntimeError("Слишком мало общих наблюдений по всем акциям.")

    data_end = pd.Timestamp(complete_returns["trading_date"].max())

    # Начинаем только после накопления минимальной истории для всех основных расчётов.
    if len(complete_returns) < MIN_COVARIANCE_OBS:
        raise RuntimeError(
            f"Нужно минимум {MIN_COVARIANCE_OBS} общих наблюдений для ковариации, "
            f"получено {len(complete_returns)}."
        )
    earliest_covariance_date = pd.Timestamp(
        complete_returns.iloc[MIN_COVARIANCE_OBS - 1]["trading_date"]
    )
    valid_factor_dates = factors.dropna(subset=FACTOR_COLUMNS + ["rf"])["trading_date"]
    if len(valid_factor_dates) < MIN_REGRESSION_OBS:
        raise RuntimeError(
            f"Нужно минимум {MIN_REGRESSION_OBS} валидных FF5-наблюдений, "
            f"получено {len(valid_factor_dates)}."
        )
    earliest_factor_date = pd.Timestamp(valid_factor_dates.iloc[MIN_REGRESSION_OBS - 1])

    valid_cap_dates = fundamentals.loc[
        fundamentals["market_cap"].notna() & (fundamentals["market_cap"] > 0),
        "effective_date",
    ].dropna().sort_values()
    if valid_cap_dates.empty:
        raise RuntimeError("Нет положительных market_cap для построения рыночных весов.")
    earliest_fundamental_date = pd.Timestamp(valid_cap_dates.iloc[0])

    eligible_ml_dates = []
    for ticker in tickers:
        stock = ml_data[ml_data["ticker"] == ticker].sort_values("trading_date")
        labelled = stock.dropna(subset=ML_FEATURES + ["target"])
        feature_dates = stock.dropna(subset=ML_FEATURES)["trading_date"]
        if len(labelled) < MIN_ML_TRAIN or feature_dates.empty:
            raise RuntimeError(
                f"Недостаточно данных для XGBoost у {ticker}: "
                f"{len(labelled)} валидных обучающих строк."
            )
        earliest_label_date = pd.Timestamp(labelled.iloc[MIN_ML_TRAIN - 1]["trading_date"])
        later_features = feature_dates[feature_dates > earliest_label_date]
        if later_features.empty:
            raise RuntimeError(f"Нет даты признаков после минимального обучения XGBoost для {ticker}.")
        eligible_ml_dates.append(pd.Timestamp(later_features.iloc[0]))
    earliest_ml_date = max(eligible_ml_dates)

    earliest_valid_date = max(
        earliest_covariance_date,
        earliest_factor_date,
        earliest_fundamental_date,
        earliest_ml_date,
    )
    if START_BACKTEST_DATE is not None:
        earliest_valid_date = max(earliest_valid_date, pd.Timestamp(START_BACKTEST_DATE))

    boundaries = [
        d for d in build_rebalance_dates(factors, imoex, earliest_valid_date)
        if d <= data_end
    ]
    if not boundaries or boundaries[0] >= data_end:
        raise RuntimeError("Недостаточно дат для walk-forward тестирования.")
    if boundaries[-1] < data_end:
        boundaries.append(data_end)
    boundaries = sorted(set(pd.Timestamp(d) for d in boundaries))
    rebalance_dates = boundaries[:-1]
    if len(rebalance_dates) < 3:
        raise RuntimeError("Недостаточно дат ребаланса после ограничения общей историей.")

    print(f"Универсум: {len(tickers)} акций")
    print(f"Общих торговых дней: {len(complete_returns)}")
    print(f"Самая ранняя дата с достаточной историей: {earliest_valid_date.date()}")
    print(f"Ковариация: до {COVARIANCE_WINDOW} общих наблюдений, минимум {MIN_COVARIANCE_OBS} с Ledoit-Wolf")
    print(f"Период тестирования: {rebalance_dates[0].date()} → {boundaries[-1].date()}")
    expected_test_days = int((complete_returns["trading_date"] > rebalance_dates[0]).sum())
    print(f"Общих дневных доходностей в тестовом окне: {expected_test_days}")
    print(f"Ребалансов: {len(rebalance_dates)}")
    print(f"Максимальный вес акции: {MAX_WEIGHT:.0%} | Проверяемые лимиты RF: "
          f"{', '.join(f'{cap:.0%}' for cap in RF_CAPS)} и свободный")
    print("ARIMA: однодневный прогноз переводится в ограниченную поправку до ±5 п.п. к годовому RF")
    print("Sharpe: средняя дневная избыточная доходность / дневная волатильность × √252")

    strategy_names = []
    for prefix, free_name in [
        ("BL", "BL_FREE_RF"),
        ("BL_ARIMA", "BL_ARIMA_FREE_RF"),
        ("BL_ARIMA_XGB", "BL_ARIMA_XGB_FREE_RF"),
    ]:
        strategy_names.append(free_name)
        strategy_names.extend(f"{prefix}_RF{int(round(cap * 100))}" for cap in RF_CAPS)
    strategy_names.extend(["EQUAL_WEIGHT_STOCKS", "RF_ONLY"])
    strategy_results = {name: [] for name in strategy_names}

    for i, date in enumerate(rebalance_dates, 1):
        print(f"[{i}/{len(rebalance_dates)}] {date.date()}")
        try:
            date_results = calculate_rebalance_portfolios(
                date, returns, factors, fundamentals, ml_data, tickers
            )
            for name in strategy_results:
                strategy_results[name].append(date_results[name])
            main_result = date_results["BL_ARIMA_XGB_RF30"]
            top = main_result.nlargest(3, "weight")
            print(
                f"  BL+ARIMA+XGB RF30: stocks={fmt_pct(main_result['weight'].sum())}, "
                f"RF={fmt_pct(main_result['risk_free_weight'].iloc[0])}; "
                + ", ".join(f"{r.ticker} {fmt_pct(r.weight)}" for r in top.itertuples())
            )
        except Exception as exc:
            raise RuntimeError(f"Ошибка на ребалансе {date.date()}: {exc}") from exc

    all_daily = []
    all_weights = []
    metrics_by_strategy = {}
    summary_rows = []
    for name, results_for_strategy in strategy_results.items():
        daily = calculate_backtest_daily(
            complete_returns, results_for_strategy, boundaries, factors, imoex
        )
        daily["portfolio_name"] = name
        metrics = calculate_backtest_metrics(daily)
        metrics_by_strategy[name] = metrics
        all_daily.append(daily)
        all_weights.append(pd.concat(results_for_strategy, ignore_index=True))
        summary_rows.append({
            "strategy": name,
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "beta_imoex": metrics["beta_imoex"],
            "tracking_error": metrics["tracking_error"],
            "avg_rf_weight": metrics["avg_rf_weight"],
            "turnover": metrics["total_turnover"],
            "transaction_cost": metrics["total_transaction_cost"],
            "benchmark_total_return": metrics["benchmark_total_return"],
        })

    daily_all = pd.concat(all_daily, ignore_index=True)
    weights_all = pd.concat(all_weights, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    print("\nСРАВНЕНИЕ СТРАТЕГИЙ")
    print(summary.to_string(index=False, formatters={
        "total_return": fmt_pct,
        "annual_return": fmt_pct,
        "volatility": fmt_pct,
        "sharpe": fmt_num,
        "max_drawdown": fmt_pct,
        "beta_imoex": fmt_num,
        "tracking_error": fmt_pct,
        "avg_rf_weight": fmt_pct,
        "turnover": fmt_pct,
        "transaction_cost": fmt_pct,
        "benchmark_total_return": fmt_pct,
    }))

    main_daily = daily_all[daily_all["portfolio_name"] == "BL_ARIMA_XGB_RF30"]
    main_metrics = metrics_by_strategy["BL_ARIMA_XGB_RF30"]
    print("\nОСНОВНОЙ ПОРТФЕЛЬ BL + ARIMA + XGBoost + RF (RF <= 30%)")
    print(f"Период: {main_daily['trading_date'].min().date()} → {main_daily['trading_date'].max().date()}")
    print(f"Торговых дней: {main_metrics['observations']} | Ребалансов: {main_metrics['rebalances']}")
    print(f"Total Return: {fmt_pct(main_metrics['total_return'])}")
    print(f"Annual Return: {fmt_pct(main_metrics['annual_return'])}")
    print(f"Volatility: {fmt_pct(main_metrics['volatility'])}")
    print(f"Sharpe: {fmt_num(main_metrics['sharpe'])}")
    print(f"Max Drawdown: {fmt_pct(main_metrics['max_drawdown'])}")
    print(f"VaR 95%: {fmt_pct(main_metrics['var_95'])} | CVaR 95%: {fmt_pct(main_metrics['cvar_95'])}")
    print(f"Beta к IMOEX: {fmt_num(main_metrics['beta_imoex'])} | Tracking Error: {fmt_pct(main_metrics['tracking_error'])}")
    print(f"Средний вес RF: {fmt_pct(main_metrics['avg_rf_weight'])}")
    print(f"IMOEX Total Return: {fmt_pct(main_metrics['benchmark_total_return'])}")
    print(f"Оборот: {fmt_pct(main_metrics['total_turnover'])} | Издержки: {fmt_pct(main_metrics['total_transaction_cost'])}")

    save_backtest(weights_all, daily_all, metrics_by_strategy)
    print("\nСохранение в БД: OK")
    print("  mart.backtest_weights — веса и прогнозные сигналы для всех стратегий")
    print("  mart.backtest_daily — дневные результаты всех стратегий")
    print("  mart.backtest_metrics — сравнительные метрики всех стратегий")


if __name__ == "__main__":
    main()
