import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

from db import get_connection

TRADING_DAYS = 252
RISK_AVERSION = 2.5
COVARIANCE_WINDOW = 252
MAX_WEIGHT = 0.10
MAX_RF_WEIGHT = 0.30
ARIMA_WINDOW = 500
ARIMA_ORDER = (1, 0, 1)
ARIMA_SIGNAL_SHRINK = 0.05
ARIMA_SIGNAL_CAP = 0.05
XGB_SIGNAL_SCALE = 0.05
MIN_ML_TRAIN = 180
RANDOM_SEED = 42
SHARPE_EPS = 1e-12
CVAR_ALPHA = 0.95
BL_WEIGHT = 0.50
ARIMA_WEIGHT = 0.30
XGB_WEIGHT = 0.20

ML_FEATURES = [
    "return_1d", "return_3d", "return_5d", "volatility_5", "volatility_20",
    "ma_ratio_5", "ma_ratio_20",
]

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

SQL_BL = """
SELECT ticker, rf_annual, market_weight, posterior_return
FROM mart.black_litterman
WHERE ticker IS NOT NULL
ORDER BY ticker
"""

SQL_PRICES = """
SELECT i.ticker, mp.trading_date, mp.close
FROM core.market_prices mp
JOIN core.instruments i ON i.instrument_id = mp.instrument_id
WHERE mp.close IS NOT NULL
ORDER BY mp.trading_date, i.ticker
"""

SQL_ML_DATA = """
SELECT ticker, trading_date, close, volume, usd_rate, gold_price, key_rate
FROM mart.market_daily
WHERE ticker IS NOT NULL AND trading_date IS NOT NULL
ORDER BY trading_date, ticker
"""

SQL_FACTORS = """
SELECT trading_date, rf
FROM mart.ff5_factors
WHERE trading_date IS NOT NULL
ORDER BY trading_date
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
        columns = [item[0] for item in cur.description]
    return pd.DataFrame(rows, columns=columns)


def to_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def db_float(value):
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def db_int(value):
    return None if value is None or pd.isna(value) else int(value)


def fmt_pct(value):
    return "N/A" if value is None or pd.isna(value) else f"{float(value) * 100:.2f}%"


def fmt_num(value):
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.3f}"


def prepare_bl(df):
    if df.empty:
        raise RuntimeError("mart.black_litterman не содержит данных.")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    for col in ["rf_annual", "market_weight", "posterior_return"]:
        df[col] = df[col].map(to_float)
    df = df.dropna(subset=["ticker", "rf_annual", "market_weight", "posterior_return"])
    return df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


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


def prepare_factors(df):
    if df.empty:
        raise RuntimeError("mart.ff5_factors не содержит данных.")
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df["rf"] = df["rf"].map(to_float)
    return (
        df.dropna(subset=["trading_date"])
        .sort_values("trading_date")
        .drop_duplicates("trading_date", keep="last")
        .reset_index(drop=True)
    )


def prepare_imoex(df):
    if df.empty:
        raise RuntimeError("IMOEX не содержит данных.")
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df["close"] = df["close"].map(to_float)
    df = df.dropna(subset=["trading_date", "close"])
    return (
        df[df["close"] > 0]
        .sort_values("trading_date")
        .drop_duplicates("trading_date", keep="last")
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
    g = df.groupby("ticker", sort=False)["close"]
    df["return_1d"] = g.pct_change(1)
    df["return_3d"] = g.pct_change(3)
    df["return_5d"] = g.pct_change(5)
    df["volatility_5"] = df.groupby("ticker", sort=False)["return_1d"].transform(
        lambda x: x.rolling(5).std()
    )
    df["volatility_20"] = df.groupby("ticker", sort=False)["return_1d"].transform(
        lambda x: x.rolling(20).std()
    )
    ma5 = g.transform(lambda x: x.rolling(5).mean())
    ma20 = g.transform(lambda x: x.rolling(20).mean())
    df["ma_ratio_5"] = df["close"] / ma5 - 1.0
    df["ma_ratio_20"] = df["close"] / ma20 - 1.0
    df["target_return"] = df.groupby("ticker", sort=False)["close"].shift(-1) / df["close"] - 1.0
    df["target"] = np.nan
    labelled = df["target_return"].notna()
    df.loc[labelled, "target"] = (df.loc[labelled, "target_return"] > 0).astype(int)
    return df


def calculate_current_rf(factors, as_of_date):
    history = factors[
        (factors["trading_date"] <= pd.Timestamp(as_of_date)) & factors["rf"].notna()
    ]
    if history.empty:
        raise RuntimeError(f"Нет безрисковой ставки на {pd.Timestamp(as_of_date).date()}.")
    daily_rf = float(history.iloc[-1]["rf"])
    if daily_rf <= -1:
        raise RuntimeError(f"Некорректная дневная RF: {daily_rf}")
    return (1.0 + daily_rf) ** TRADING_DAYS - 1.0


def build_returns(prices, tickers, as_of_date):
    prices = prices[
        (prices["ticker"].isin(tickers))
        & (prices["trading_date"] <= pd.Timestamp(as_of_date))
    ]
    pivot = prices.pivot_table(
        index="trading_date", columns="ticker", values="close", aggfunc="last"
    ).reindex(columns=tickers)
    returns = pivot.pct_change().dropna(how="any")
    if len(returns) < COVARIANCE_WINDOW:
        raise RuntimeError(f"Недостаточно общих наблюдений для ковариации: {len(returns)}.")
    return returns.tail(COVARIANCE_WINDOW)


def build_covariance(returns):
    estimator = LedoitWolf(assume_centered=False).fit(returns.to_numpy(dtype=float))
    return pd.DataFrame(
        estimator.covariance_ * TRADING_DAYS,
        index=returns.columns,
        columns=returns.columns,
    )


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
            continue
    return None


def predict_arima(ml_data, ticker, as_of_date, rf_annual):
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

    # Прогноз относится к одному дню. Он не считается доходностью,
    # которая автоматически сохраняется все 252 торговых дня.
    raw_linear_annual = prediction * TRADING_DAYS
    alpha_annual = float(np.clip(
        raw_linear_annual * ARIMA_SIGNAL_SHRINK,
        -ARIMA_SIGNAL_CAP,
        ARIMA_SIGNAL_CAP,
    ))
    return prediction, float(raw_linear_annual), alpha_annual, float(rf_annual + alpha_annual)


def predict_xgb(ml_data, ticker, as_of_date, rf_annual):
    stock = ml_data[ml_data["ticker"] == ticker].sort_values("trading_date").copy()
    train = stock[stock["trading_date"] < pd.Timestamp(as_of_date)].dropna(
        subset=ML_FEATURES + ["target"]
    )
    current = stock[stock["trading_date"] <= pd.Timestamp(as_of_date)].dropna(
        subset=ML_FEATURES
    )
    if len(train) < MIN_ML_TRAIN:
        raise RuntimeError(f"Для {ticker} только {len(train)} обучающих строк XGBoost.")
    if current.empty:
        raise RuntimeError(f"Нет признаков XGBoost для {ticker}.")
    train = train.tail(756)
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
    probabilities = model.predict_proba(row[ML_FEATURES].astype(float))[0]
    classes = list(model.classes_)
    p_up = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    p_up = float(np.clip(p_up, 0.0, 1.0))
    alpha_annual = XGB_SIGNAL_SCALE * (2.0 * p_up - 1.0)
    return p_up, float(alpha_annual), float(rf_annual + alpha_annual), int(len(train)), pd.Timestamp(row["trading_date"].iloc[0]).date()


def calculate_signals(ml_data, tickers, as_of_date, rf_annual):
    rows = []
    for ticker in tickers:
        arima_daily, arima_raw_annual, arima_alpha_annual, arima_expected = predict_arima(
            ml_data, ticker, as_of_date, rf_annual
        )
        p_up, xgb_alpha, xgb_expected, train_rows, feature_date = predict_xgb(
            ml_data, ticker, as_of_date, rf_annual
        )
        rows.append({
            "ticker": ticker,
            "arima_pred_daily": arima_daily,
            "arima_pred_annual_raw": arima_raw_annual,
            "arima_alpha_annual": arima_alpha_annual,
            "arima_expected_return_annual": arima_expected,
            "xgb_probability_growth": p_up,
            "xgb_probability_down": 1.0 - p_up,
            "xgb_signal_alpha_annual": xgb_alpha,
            "xgb_expected_return_annual": xgb_expected,
            "ml_train_rows": train_rows,
            "ml_feature_date": feature_date,
        })
    return pd.DataFrame(rows)


def optimize_portfolio(expected_returns, covariance, rf_annual, initial_weights, max_rf_weight):
    n = len(expected_returns)
    minimum_stock_weight = 1.0 - max_rf_weight

    def utility(weights):
        expected_total = rf_annual + weights @ (expected_returns - rf_annual)
        variance = max(float(weights @ covariance @ weights), 0.0)
        return -(expected_total - RISK_AVERSION / 2.0 * variance)

    constraints = [
        {"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)},
        {"type": "ineq", "fun": lambda w: np.sum(w) - minimum_stock_weight},
    ]
    result = minimize(
        utility,
        np.asarray(initial_weights, dtype=float),
        method="SLSQP",
        bounds=[(0.0, MAX_WEIGHT)] * n,
        constraints=constraints,
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
    if weights.max(initial=0.0) > MAX_WEIGHT + 1e-8:
        raise RuntimeError("Нарушено ограничение веса одной акции.")
    return weights, max(0.0, 1.0 - stock_sum), result


def calculate_var_cvar(returns, alpha=CVAR_ALPHA):
    values = pd.Series(returns).dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    losses = -values
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    scale = math.sqrt(TRADING_DAYS)
    return var * scale, (float(tail.mean()) if len(tail) else var) * scale


def calculate_max_drawdown(returns):
    curve = (1.0 + pd.Series(returns).dropna().astype(float)).cumprod()
    return float((curve / curve.cummax() - 1.0).min()) if not curve.empty else np.nan


def calculate_tracking_error(portfolio, benchmark):
    p, b = portfolio.align(benchmark, join="inner")
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1, sort=False).dropna()
    if len(df) < 2:
        return np.nan
    return float((df["p"] - df["b"]).std(ddof=1) * math.sqrt(TRADING_DAYS))


def calculate_beta(portfolio, benchmark):
    p, b = portfolio.align(benchmark, join="inner")
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1, sort=False).dropna()
    if len(df) < 2:
        return np.nan
    market_variance = df["b"].var(ddof=1)
    if pd.isna(market_variance) or market_variance <= 0:
        return np.nan
    return float(np.cov(df["p"], df["b"], ddof=1)[0, 1] / market_variance)


def annualized_return(returns):
    values = pd.Series(returns).dropna().astype(float)
    if values.empty:
        return np.nan
    growth = float((1.0 + values).prod())
    years = len(values) / TRADING_DAYS
    return float(growth ** (1.0 / years) - 1.0) if growth > 0 and years > 0 else np.nan


def calculate_metrics(weights, rf_weight, expected_returns, covariance, returns, rf_daily, imoex, current_rf):
    stock_daily = returns.mul(weights, axis=1).sum(axis=1)
    daily = pd.concat([stock_daily.rename("stock"), rf_daily.rename("rf")], axis=1, sort=False).dropna()
    portfolio = daily["stock"] + rf_weight * daily["rf"]
    benchmark = imoex.set_index("trading_date")["close"].pct_change()

    model_expected = float(current_rf + np.asarray(weights) @ (np.asarray(expected_returns) - current_rf))
    model_volatility = math.sqrt(max(float(np.asarray(weights) @ np.asarray(covariance) @ np.asarray(weights)), 0.0))
    historical_volatility = float(portfolio.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(portfolio) > 1 else np.nan
    sharpe = (model_expected - current_rf) / model_volatility if model_volatility > SHARPE_EPS else np.nan

    observed_rf = daily["rf"].astype(float)
    historical_rf = float((1.0 + observed_rf.mean()) ** TRADING_DAYS - 1.0) if not observed_rf.empty else np.nan
    historical_return = annualized_return(portfolio)
    historical_sharpe = (
        (historical_return - historical_rf) / historical_volatility
        if historical_volatility > SHARPE_EPS and pd.notna(historical_rf) else np.nan
    )
    var_95, cvar_95 = calculate_var_cvar(portfolio)
    return {
        "expected_return": model_expected,
        "volatility": model_volatility,
        "sharpe": sharpe,
        "historical_return": historical_return,
        "historical_volatility": historical_volatility,
        "historical_sharpe": historical_sharpe,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_drawdown": calculate_max_drawdown(portfolio),
        "tracking_error": calculate_tracking_error(portfolio, benchmark),
        "beta_imoex": calculate_beta(portfolio, benchmark),
        "observations": len(portfolio),
        "rf_annual": float(current_rf),
    }


CREATE_WEIGHTS_SQL = """
DROP TABLE IF EXISTS mart.portfolio_weights;
CREATE TABLE mart.portfolio_weights (
    portfolio_name VARCHAR(60) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    calc_date DATE NOT NULL,
    method VARCHAR(50) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    risk_free_weight DOUBLE PRECISION NOT NULL,
    market_weight DOUBLE PRECISION NOT NULL,
    posterior_return DOUBLE PRECISION NOT NULL,
    arima_pred_daily DOUBLE PRECISION NOT NULL,
    arima_pred_annual_raw DOUBLE PRECISION NOT NULL,
    arima_alpha_annual DOUBLE PRECISION NOT NULL,
    arima_expected_return_annual DOUBLE PRECISION NOT NULL,
    xgb_probability_growth DOUBLE PRECISION NOT NULL,
    xgb_probability_down DOUBLE PRECISION NOT NULL,
    xgb_signal_alpha_annual DOUBLE PRECISION NOT NULL,
    xgb_expected_return_annual DOUBLE PRECISION NOT NULL,
    expected_return_annual DOUBLE PRECISION NOT NULL,
    contribution_expected_return DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (portfolio_name, ticker)
);
"""

CREATE_METRICS_SQL = """
DROP TABLE IF EXISTS mart.portfolio_metrics;
CREATE TABLE mart.portfolio_metrics (
    portfolio_name VARCHAR(60) PRIMARY KEY,
    calc_date DATE NOT NULL,
    method VARCHAR(50) NOT NULL,
    observations INTEGER NOT NULL,
    expected_return_annual DOUBLE PRECISION NOT NULL,
    volatility_annual DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION,
    historical_return_annual DOUBLE PRECISION,
    historical_volatility_annual DOUBLE PRECISION,
    historical_sharpe_ratio DOUBLE PRECISION,
    var_95_annual_approx DOUBLE PRECISION,
    cvar_95_annual_approx DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    tracking_error_annual DOUBLE PRECISION,
    beta_imoex DOUBLE PRECISION,
    risk_free_rate DOUBLE PRECISION NOT NULL,
    risk_free_weight DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def save_results(weight_rows, metric_rows):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_WEIGHTS_SQL)
            cur.execute(CREATE_METRICS_SQL)
            weight_cols = [
                "portfolio_name", "ticker", "calc_date", "method", "weight", "risk_free_weight",
                "market_weight", "posterior_return", "arima_pred_daily", "arima_pred_annual_raw",
                "arima_alpha_annual", "arima_expected_return_annual", "xgb_probability_growth",
                "xgb_probability_down", "xgb_signal_alpha_annual", "xgb_expected_return_annual",
                "expected_return_annual", "contribution_expected_return",
            ]
            rows = []
            for _, row in weight_rows.iterrows():
                rows.append(tuple(
                    row[col] if col in {"portfolio_name", "ticker", "calc_date", "method"}
                    else db_float(row[col])
                    for col in weight_cols
                ))
            cur.executemany(
                f"INSERT INTO mart.portfolio_weights ({','.join(weight_cols)}) VALUES ({','.join(['%s'] * len(weight_cols))})",
                rows,
            )
            metric_cols = [
                "portfolio_name", "calc_date", "method", "observations", "expected_return_annual",
                "volatility_annual", "sharpe_ratio", "historical_return_annual",
                "historical_volatility_annual", "historical_sharpe_ratio", "var_95_annual_approx",
                "cvar_95_annual_approx", "max_drawdown", "tracking_error_annual", "beta_imoex",
                "risk_free_rate", "risk_free_weight",
            ]
            metric_values = []
            for name, metrics in metric_rows.items():
                metric_values.append((
                    name, metrics["calc_date"], metrics["method"], db_int(metrics["observations"]),
                    db_float(metrics["expected_return"]), db_float(metrics["volatility"]),
                    db_float(metrics["sharpe"]), db_float(metrics["historical_return"]),
                    db_float(metrics["historical_volatility"]), db_float(metrics["historical_sharpe"]),
                    db_float(metrics["var_95"]), db_float(metrics["cvar_95"]),
                    db_float(metrics["max_drawdown"]), db_float(metrics["tracking_error"]),
                    db_float(metrics["beta_imoex"]), db_float(metrics["rf_annual"]),
                    db_float(metrics["risk_free_weight"]),
                ))
            cur.executemany(
                f"INSERT INTO mart.portfolio_metrics ({','.join(metric_cols)}) VALUES ({','.join(['%s'] * len(metric_cols))})",
                metric_values,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    print("ОПТИМИЗАЦИЯ И СРАВНЕНИЕ ПОРТФЕЛЕЙ\n")
    conn = get_connection()
    try:
        bl = read_sql_df(SQL_BL, conn)
        prices = read_sql_df(SQL_PRICES, conn)
        ml_raw = read_sql_df(SQL_ML_DATA, conn)
        factors = read_sql_df(SQL_FACTORS, conn)
        imoex = read_sql_df(SQL_IMOEX, conn)
    finally:
        conn.close()

    bl = prepare_bl(bl)
    prices = prepare_prices(prices)
    ml_data = prepare_ml_data(ml_raw)
    factors = prepare_factors(factors)
    imoex = prepare_imoex(imoex)
    tickers = bl["ticker"].tolist()

    dates = [
        prices["trading_date"].max(),
        ml_data["trading_date"].max(),
        factors["trading_date"].max(),
        imoex["trading_date"].max(),
    ]
    as_of_date = min(pd.Timestamp(date) for date in dates if pd.notna(date))
    rf = calculate_current_rf(factors, as_of_date)
    returns = build_returns(prices, tickers, as_of_date)
    covariance = build_covariance(returns)
    signals = calculate_signals(ml_data, tickers, as_of_date, rf)
    expected = bl.merge(signals, on="ticker", how="inner", validate="one_to_one")
    expected = expected.set_index("ticker").loc[tickers].reset_index()

    # Прогноз одного дня используется как ограниченная поправка к RF,
    # а не как предположение, что такой темп доходности сохранится 252 дня.
    bl_return = expected["posterior_return"].astype(float).to_numpy()
    arima_return = expected["arima_expected_return_annual"].astype(float).to_numpy()
    xgb_return = expected["xgb_expected_return_annual"].astype(float).to_numpy()

    strategy_configs = {
        "BL_FREE_RF": (bl_return, 1.00),
        "BL_RF30": (bl_return, MAX_RF_WEIGHT),
        "BL_ARIMA_FREE_RF": (0.70 * bl_return + 0.30 * arima_return, 1.00),
        "BL_ARIMA_RF30": (0.70 * bl_return + 0.30 * arima_return, MAX_RF_WEIGHT),
        "BL_ARIMA_XGB_FREE_RF": (BL_WEIGHT * bl_return + ARIMA_WEIGHT * arima_return + XGB_WEIGHT * xgb_return, 1.00),
        "BL_ARIMA_XGB_RF30": (BL_WEIGHT * bl_return + ARIMA_WEIGHT * arima_return + XGB_WEIGHT * xgb_return, MAX_RF_WEIGHT),
    }

    rf_series = (
        factors[factors["rf"].notna()]
        .sort_values("trading_date")
        .drop_duplicates("trading_date", keep="last")
        .set_index("trading_date")["rf"]
        .reindex(returns.index)
    )
    all_weights = []
    metric_rows = {}
    strategy_summary = []

    for strategy_name, (expected_returns, max_rf_weight) in strategy_configs.items():
        initial = np.repeat(1.0 / len(tickers), len(tickers))
        weights, rf_weight, opt = optimize_portfolio(
            expected_returns, covariance.to_numpy(), rf, initial, max_rf_weight
        )
        result = expected.copy()
        result["portfolio_name"] = strategy_name
        result["method"] = "UTILITY_BL_ARIMA_XGB"
        result["weight"] = weights
        result["risk_free_weight"] = rf_weight
        result["expected_return_annual"] = expected_returns
        result["contribution_expected_return"] = result["weight"] * expected_returns
        result["calc_date"] = as_of_date.date()
        all_weights.append(result)

        metrics = calculate_metrics(
            weights, rf_weight, expected_returns, covariance.to_numpy(),
            returns, rf_series, imoex, rf,
        )
        metrics["calc_date"] = as_of_date.date()
        metrics["method"] = "UTILITY_BL_ARIMA_XGB"
        metrics["risk_free_weight"] = rf_weight
        metric_rows[strategy_name] = metrics
        strategy_summary.append({
            "strategy": strategy_name,
            "expected_return": metrics["expected_return"],
            "expected_volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"],
            "historical_return": metrics["historical_return"],
            "historical_volatility": metrics["historical_volatility"],
            "max_drawdown": metrics["max_drawdown"],
            "rf_weight": rf_weight,
            "max_stock_weight": float(weights.max(initial=0.0)),
            "solver": str(opt.message),
        })

    summary = pd.DataFrame(strategy_summary)
    print(f"Дата расчёта: {as_of_date.date()} | Акций: {len(tickers)} | RF: {fmt_pct(rf)}")
    print(f"Ограничение одной акции: {MAX_WEIGHT:.0%} | Основное ограничение RF: не более {MAX_RF_WEIGHT:.0%}\n")
    print("СРАВНЕНИЕ СТРАТЕГИЙ")
    print(summary[["strategy", "expected_return", "expected_volatility", "sharpe", "historical_return", "historical_volatility", "max_drawdown", "rf_weight"]].to_string(
        index=False,
        formatters={
            "expected_return": fmt_pct,
            "expected_volatility": fmt_pct,
            "sharpe": fmt_num,
            "historical_return": fmt_pct,
            "historical_volatility": fmt_pct,
            "max_drawdown": fmt_pct,
            "rf_weight": fmt_pct,
        },
    ))

    main_name = "BL_ARIMA_XGB_RF30"
    main_result = next(frame for frame in all_weights if frame["portfolio_name"].iloc[0] == main_name)
    main_result = main_result.sort_values("weight", ascending=False)
    print(f"\nОСНОВНОЙ ПОРТФЕЛЬ: {main_name}")
    for r in main_result.itertuples():
        if r.weight > 1e-8:
            print(
                f"{r.ticker:<6} {fmt_pct(r.weight):>8} BL={fmt_pct(r.posterior_return):>8} "
                f"ARIMA α={fmt_pct(r.arima_alpha_annual):>8} P(up)={fmt_pct(r.xgb_probability_growth):>8} "
                f"E={fmt_pct(r.expected_return_annual):>8}"
            )
    print(f"{'RF':<6} {fmt_pct(metric_rows[main_name]['risk_free_weight']):>8}")
    print(f"Сумма stock + RF: {main_result['weight'].sum() + metric_rows[main_name]['risk_free_weight']:.6f}")
    print(f"Все веса акций не выше {MAX_WEIGHT:.0%}: {(main_result['weight'].max() <= MAX_WEIGHT + 1e-8)}")

    save_results(pd.concat(all_weights, ignore_index=True), metric_rows)
    print("\nСохранение в БД: OK")
    print("  mart.portfolio_weights")
    print("  mart.portfolio_metrics")


if __name__ == "__main__":
    main()
