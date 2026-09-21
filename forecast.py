import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from db import get_connection


FEATURES = [
    "return_1d", "return_3d", "return_5d", "volatility_5", "volatility_20",
    "ma_ratio_5", "ma_ratio_20", "usd_return_1d", "gold_return_1d",
    "key_rate", "volume_ratio_5"
]

PERIODS = {
    "2020-2026": "2020-01-01",
    "2022-2026": "2022-01-01",
    "2024-2026": "2024-01-01",
    "2025-2026": "2025-01-01"
}

TEST_SIZE = 0.2
ARIMA_ORDER = (1, 0, 1)
ARIMA_WINDOW = 500
ARIMA_REFIT_STEP = 5
MAX_WORKERS = 6
RANDOM_SEED = 42


def load_data():
    conn = get_connection()
    query = """
        SELECT ticker, trading_date, close, volume,
               usd_rate, eur_rate, gold_price, key_rate
        FROM mart.market_daily
        ORDER BY ticker, trading_date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df.sort_values(["ticker", "trading_date"]).reset_index(drop=True)


def prepare_data(df):
    df = df.sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    g = df.groupby("ticker")

    df["return_1d"] = g["close"].pct_change()
    df["return_3d"] = g["close"].pct_change(3)
    df["return_5d"] = g["close"].pct_change(5)

    df["volatility_5"] = g["return_1d"].transform(lambda x: x.rolling(5).std())
    df["volatility_20"] = g["return_1d"].transform(lambda x: x.rolling(20).std())

    ma5 = g["close"].transform(lambda x: x.rolling(5).mean())
    ma20 = g["close"].transform(lambda x: x.rolling(20).mean())
    df["ma_ratio_5"] = df["close"] / ma5 - 1
    df["ma_ratio_20"] = df["close"] / ma20 - 1

    vol5 = g["volume"].transform(lambda x: x.rolling(5).mean())
    df["volume_ratio_5"] = df["volume"] / vol5

    macro = df[["trading_date", "usd_rate", "gold_price"]].drop_duplicates("trading_date")
    macro = macro.sort_values("trading_date")
    macro["usd_return_1d"] = macro["usd_rate"].pct_change()
    macro["gold_return_1d"] = macro["gold_price"].pct_change()

    df = df.merge(
        macro[["trading_date", "usd_return_1d", "gold_return_1d"]],
        on="trading_date", how="left"
    )

    df["target_return"] = g["close"].shift(-1) / df["close"] - 1
    return df


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mask = np.isfinite(y) & np.isfinite(pred)
    direction = np.mean(np.sign(y[mask]) == np.sign(pred[mask])) * 100 if mask.any() else np.nan
    return mae, rmse, direction


def fit_arima(history):
    history = np.asarray(history, float)
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
                    enforce_invertibility=True
                ).fit(method_kwargs={"maxiter": 200})
        except Exception:
            pass

    return None


def rolling_arima(train, test):
    history = list(train)
    pred = []
    model = fit_arima(history)

    if model is None:
        return np.full(len(test), history[-1] if history else 0.0)

    for i, actual in enumerate(test):
        try:
            p = float(model.forecast(1)[0])
        except Exception:
            p = history[-1]

        if not np.isfinite(p):
            p = history[-1]

        pred.append(p)
        history.append(actual)

        try:
            refit = (i + 1) % ARIMA_REFIT_STEP == 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model.append([actual], refit=refit)
        except Exception:
            model = fit_arima(history)
            if model is None:
                break

    if len(pred) < len(test):
        pred += [history[-1]] * (len(test) - len(pred))

    return np.asarray(pred)


def train_catboost(X_train, y_train, X_test, iterations=500):
    model = CatBoostRegressor(
        iterations=iterations, depth=6, learning_rate=0.05,
        loss_function="RMSE", verbose=False,
        random_seed=RANDOM_SEED, thread_count=1,
        allow_writing_files=False
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def process_period(df, period_name, period_start):
    ticker = df["ticker"].iloc[0]
    df = df[df["trading_date"] >= period_start].sort_values("trading_date").reset_index(drop=True)

    model_df = df.dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)

    if len(model_df) < 100:
        return [], ticker

    split = int(len(model_df) * (1 - TEST_SIZE))
    train, test = model_df.iloc[:split], model_df.iloc[split:]

    if len(train) < 30 or test.empty:
        return [], ticker

    X_train = train[FEATURES].astype(float).to_numpy()
    X_test = test[FEATURES].astype(float).to_numpy()
    y_train = train["target_return"].astype(float).to_numpy()
    y_test = test["target_return"].astype(float).to_numpy()

    naive = test["return_1d"].astype(float).to_numpy()
    arima = rolling_arima(y_train, y_test)
    catboost = train_catboost(X_train, y_train, X_test)

    rows = []
    for name, pred in [("Naive", naive), ("ARIMA", arima), ("CatBoost", catboost)]:
        mae, rmse, direction = metrics(y_test, pred)
        rows.append({
            "ticker": ticker,
            "period": period_name,
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy": direction,
            "train_start": train["trading_date"].min(),
            "train_end": train["trading_date"].max(),
            "test_start": test["trading_date"].min(),
            "test_end": test["trading_date"].max()
        })

    return rows, ticker


def worker(args):
    ticker, df, period_name, period_start = args
    try:
        rows, ticker = process_period(df, period_name, period_start)
        return ticker, period_name, rows, None
    except Exception as e:
        return ticker, period_name, [], str(e)


def final_forecast(df, period_name, period_start):
    ticker = df["ticker"].iloc[0]
    df = df[df["trading_date"] >= period_start].sort_values("trading_date").reset_index(drop=True)

    train = df.dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)
    final = df.dropna(subset=FEATURES).reset_index(drop=True)

    if len(train) < 100 or final.empty:
        return []

    last = final.iloc[-1]
    date = last["trading_date"]
    close = float(last["close"])

    X = train[FEATURES].astype(float).to_numpy()
    y = train["target_return"].astype(float).to_numpy()
    X_last = final[FEATURES].iloc[[-1]].astype(float).to_numpy()

    naive = float(final["return_1d"].iloc[-1])
    if not np.isfinite(naive):
        naive = float(y[-1])

    model = fit_arima(y)
    try:
        arima = float(model.forecast(1)[0]) if model else float(y[-1])
    except Exception:
        arima = float(y[-1])

    if not np.isfinite(arima):
        arima = float(y[-1])

    try:
        catboost = train_catboost(X, y, X_last, iterations=150)[0]
        catboost = float(catboost) if np.isfinite(catboost) else np.nan
    except Exception:
        catboost = np.nan

    returns = [naive, arima, catboost]
    prices = [close * (1 + x) if np.isfinite(x) else np.nan for x in returns]

    return [
        {
            "ticker": ticker,
            "period": period_name,
            "forecast_date": date,
            "model": model_name,
            "current_price": close,
            "forecast_return": ret,
            "forecast_price": price
        }
        for model_name, ret, price in zip(
            ["Naive", "ARIMA", "CatBoost"], returns, prices
        )
    ]


def fmt_results(df):
    result = df.copy()
    result["mae"] = result["mae"].map(lambda x: f"{x:.5f}")
    result["rmse"] = result["rmse"].map(lambda x: f"{x:.5f}")
    result["directional_accuracy"] = result["directional_accuracy"].map(lambda x: f"{x:.2f}%")
    return result


def main():
    df = prepare_data(load_data())
    tickers = sorted(df["ticker"].unique())
    groups = [(t, df[df["ticker"] == t].copy()) for t in tickers]

    tasks = [
        (ticker, group, period, start)
        for period, start in PERIODS.items()
        for ticker, group in groups
    ]

    print(f"Компаний: {len(tickers)}")
    print(f"Периодов: {len(PERIODS)}")
    print(f"Задач: {len(tasks)}")

    all_metrics, errors = [], []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, task) for task in tasks]

        for i, future in enumerate(as_completed(futures), 1):
            ticker, period, rows, error = future.result()
            all_metrics.extend(rows)

            if error:
                errors.append((ticker, period, error))

            print(f"\rОбработано: {i}/{len(tasks)}", end="")

    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["period", "ticker", "model"]).reset_index(drop=True)

    print("\n\n")
    print("РЕЗУЛЬТАТЫ")

    if not metrics_df.empty:
        print(fmt_results(
            metrics_df[
                ["period", "ticker", "model", "mae", "rmse", "directional_accuracy"]
            ]
        ).to_string(index=False))

    print("Средние по периодам")
    if not metrics_df.empty:
        summary = (
            metrics_df.groupby(["period", "model"], as_index=False)
            [["mae", "rmse", "directional_accuracy"]].mean()
        )
        print(fmt_results(summary).to_string(index=False))

    print("Лучшие модели MAE")

    if not metrics_df.empty:
        best = (
            metrics_df.sort_values(["period", "ticker", "mae"])
            .groupby(["period", "ticker"], as_index=False).first()
        )
        print(fmt_results(
            best[["period", "ticker", "model", "mae", "rmse", "directional_accuracy"]]
        ).to_string(index=False))

        wins = (
            best.groupby(["period", "model"])
            .size()
            .reset_index(name="wins")
        )

        print("\nПобеды:")
        print(wins.to_string(index=False))

    print("ФИНАЛЬНЫЕ ПРОГНОЗЫ")
    forecasts, forecast_errors = [], []

    for i, (ticker, group) in enumerate(
        [(t, g) for t, g in groups for _ in PERIODS], 1
    ):
        period_name = list(PERIODS)[(i - 1) % len(PERIODS)]
        period_start = PERIODS[period_name]

        try:
            forecasts.extend(final_forecast(group, period_name, period_start))
        except Exception as e:
            forecast_errors.append((ticker, period_name, str(e)))

        print(f"\rПрогнозы: {i}/{len(tasks)}", end="")

    forecasts_df = pd.DataFrame(forecasts)

    if not forecasts_df.empty:
        forecasts_df = forecasts_df.sort_values(
            ["period", "ticker", "model"]
        ).reset_index(drop=True)

        table = forecasts_df[
            [
                "period", "ticker", "forecast_date", "model",
                "current_price", "forecast_return", "forecast_price"
            ]
        ].copy()

        table["current_price"] = table["current_price"].map(lambda x: f"{x:.2f}")
        table["forecast_return"] = table["forecast_return"].map(
            lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "N/A"
        )
        table["forecast_price"] = table["forecast_price"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

        print("\n")
        print(table.to_string(index=False))

    print("\n" + "=" * 100)
    print("ИТОГ")
    print("=" * 100)

    print(f"Компаний обработано: {metrics_df['ticker'].nunique()}/{len(tickers)}")
    print(f"Периодов обработано: {metrics_df['period'].nunique()}/{len(PERIODS)}")
    print(f"Метрик рассчитано: {len(metrics_df)}")
    print(f"Прогнозов рассчитано: {len(forecasts_df)}")
    print(f"Ошибок: {len(errors) + len(forecast_errors)}")

    for ticker, period, error in errors + forecast_errors:
        print(f"  {ticker} [{period}]: {error}")


if __name__ == "__main__":
    main()