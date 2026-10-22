import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from db import get_connection

warnings.filterwarnings("ignore")

PERIODS = {
    "2020-2026": "2020-01-01",
    "2022-2026": "2022-01-01",
    "2024-2026": "2024-01-01",
    "2025-2026": "2025-01-01",
}

TEST_SIZE = 0.20
SEARCH_SPLITS = 3
N_ITER = 12
RANDOM_STATE = 42

FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5",
    "volatility_20",
    "ma_ratio_5",
    "ma_ratio_20",
]

XGB_PARAMS = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.03, 0.05, 0.10],
    "max_depth": [2, 3, 4, 5],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0.0, 0.05, 0.10, 0.20],
    "reg_alpha": [0.0, 0.01, 0.10, 0.50],
    "reg_lambda": [1, 2, 5, 10],
}


def load_data():
    conn = get_connection()
    try:
        df = pd.read_sql(
            """
            SELECT ticker, trading_date, close, volume, usd_rate, gold_price
            FROM mart.market_daily
            ORDER BY ticker, trading_date
            """,
            conn,
        )
    finally:
        conn.close()

    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df.sort_values(["ticker", "trading_date"]).reset_index(drop=True)


def prepare_features(df):
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
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
    df["volatility_5"] = df.groupby("ticker", sort=False)["return_1d"].transform(
        lambda x: x.rolling(5).std()
    )
    df["volatility_20"] = df.groupby("ticker", sort=False)["return_1d"].transform(
        lambda x: x.rolling(20).std()
    )

    ma_5 = grouped.transform(lambda x: x.rolling(5).mean())
    ma_20 = grouped.transform(lambda x: x.rolling(20).mean())
    df["ma_ratio_5"] = df["close"] / ma_5 - 1.0
    df["ma_ratio_20"] = df["close"] / ma_20 - 1.0

    # Метка неизвестна для последней даты, пока не появится следующая цена.
    df["next_return"] = grouped.shift(-1) / df["close"] - 1.0
    df["target"] = np.nan
    labelled = df["next_return"].notna()
    df.loc[labelled, "target"] = (df.loc[labelled, "next_return"] > 0).astype(int)

    return df.dropna(subset=FEATURES).reset_index(drop=True)


def create_dummy():
    return DummyClassifier(strategy="prior")


def create_gradient_boosting():
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )


def create_xgb():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(y_true, y_pred, y_prob):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    result["roc_auc"] = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) == 2
        else np.nan
    )
    return result


def positive_probability(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = positive_probability(model, X_test)
    return calculate_metrics(y_test, predictions, probabilities)


def tune_xgboost(X_train, y_train):
    if len(np.unique(y_train)) < 2:
        return {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_alpha": 0.01,
            "reg_lambda": 5,
        }

    tscv = TimeSeriesSplit(n_splits=SEARCH_SPLITS)
    search = RandomizedSearchCV(
        estimator=create_xgb(),
        param_distributions=XGB_PARAMS,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
        error_score=np.nan,
    )
    search.fit(X_train, y_train)
    return search.best_params_


def create_tuned_xgb(params):
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    base_params.update(params)
    return XGBClassifier(**base_params)


def run_experiment(df):
    results = []
    final_forecasts = []
    best_params = []
    tickers = sorted(df["ticker"].unique())
    total_tasks = len(tickers) * len(PERIODS)
    current_task = 0

    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker].copy()
        for period, start_date in PERIODS.items():
            current_task += 1
            print(f"[{current_task}/{total_tasks}] {ticker} | {period}")

            period_df = ticker_df[
                ticker_df["trading_date"] >= pd.Timestamp(start_date)
            ].sort_values("trading_date")
            labelled = period_df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
            current = period_df.dropna(subset=FEATURES).reset_index(drop=True)

            if len(labelled) < 100 or current.empty:
                print("  пропуск: мало данных")
                continue

            split_index = int(len(labelled) * (1.0 - TEST_SIZE))
            train = labelled.iloc[:split_index]
            test = labelled.iloc[split_index:]
            if len(train) < 30 or test.empty:
                continue

            X_train = train[FEATURES].astype(float).to_numpy()
            y_train = train["target"].astype(int).to_numpy()
            X_test = test[FEATURES].astype(float).to_numpy()
            y_test = test["target"].astype(int).to_numpy()

            dummy_metrics = evaluate_model(
                create_dummy(), X_train, y_train, X_test, y_test
            )
            results.append({"ticker": ticker, "period": period, "model": "Dummy", **dummy_metrics})

            gb_metrics = evaluate_model(
                create_gradient_boosting(), X_train, y_train, X_test, y_test
            )
            results.append({"ticker": ticker, "period": period, "model": "GradientBoosting", **gb_metrics})

            best_xgb_params = tune_xgboost(X_train, y_train)
            best_params.append({"ticker": ticker, "period": period, **best_xgb_params})

            xgb_metrics = evaluate_model(
                create_tuned_xgb(best_xgb_params), X_train, y_train, X_test, y_test
            )
            results.append({"ticker": ticker, "period": period, "model": "XGBoost", **xgb_metrics})

            all_X = labelled[FEATURES].astype(float).to_numpy()
            all_y = labelled["target"].astype(int).to_numpy()
            final_model = create_tuned_xgb(best_xgb_params)
            final_model.fit(all_X, all_y)

            last_row = current.iloc[[-1]]
            X_last = last_row[FEATURES].astype(float)
            prediction = int(final_model.predict(X_last)[0])
            probability_growth = float(positive_probability(final_model, X_last)[0])
            final_forecasts.append({
                "ticker": ticker,
                "period": period,
                "model": "XGBoost",
                "forecast_date": last_row["trading_date"].iloc[0],
                "close": last_row["close"].iloc[0],
                "prediction": "GROWTH" if prediction == 1 else "DOWN",
                "probability_growth": f"{probability_growth:.2%}",
                "probability_down": f"{1.0 - probability_growth:.2%}",
            })

    return pd.DataFrame(results), pd.DataFrame(final_forecasts), pd.DataFrame(best_params)


def print_summary(results):
    print("Средние результаты TEST")
    summary = (
        results.groupby(["period", "model"])[
            ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
        ]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False, formatters={
        "accuracy": "{:.2%}".format,
        "balanced_accuracy": "{:.2%}".format,
        "precision": "{:.2%}".format,
        "recall": "{:.2%}".format,
        "f1": "{:.2%}".format,
        "roc_auc": "{:.3f}".format,
    }))

    print("\nПобедители по ACCURACY")
    winners = []
    for period in PERIODS:
        scores = results[results["period"] == period].groupby("model")["accuracy"].mean()
        if not scores.empty:
            winners.append({"period": period, "winner": scores.idxmax(), "accuracy": scores.max()})
    print(pd.DataFrame(winners).to_string(index=False, formatters={"accuracy": "{:.2%}".format}))

    wins = 0
    total = 0
    for _, group in results.groupby(["ticker", "period"]):
        if not group.empty:
            total += 1
            wins += int(group.loc[group["accuracy"].idxmax(), "model"] == "XGBoost")
    print(f"\nXGBoost — лучший по Accuracy в {wins} из {total} экспериментов ({wins / total:.2%})")


def print_forecasts(forecasts):
    print("\nФИНАЛЬНЫЕ ПРОГНОЗЫ")
    print(forecasts.to_string(index=False))


def print_best_params(params):
    print("\nЧАСТОТА ВЫБРАННЫХ ПАРАМЕТРОВ XGBOOST")
    if params.empty:
        print("Параметры не рассчитаны.")
        return
    for column in [
        "n_estimators", "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
    ]:
        if column in params:
            print(f"\n{column}:")
            print(params[column].value_counts().head(5).to_string())


def main():
    print("ЗАГРУЗКА ДАННЫХ")
    df = load_data()
    print(f"Загружено строк: {len(df):,}")
    df = prepare_features(df)
    print(f"После подготовки признаков: {len(df):,}")

    print("\nЗАПУСК ЭКСПЕРИМЕНТА")
    results, forecasts, params = run_experiment(df)
    if results.empty:
        raise RuntimeError("Не получено ни одной оценки классификатора.")

    print_summary(results)
    print_forecasts(forecasts)
    print_best_params(params)
    print("\nИТОГ")
    print(f"Компаний: {results['ticker'].nunique()}/25")
    print(f"Периодов: {results['period'].nunique()}/4")
    print(f"Оценок моделей: {len(results)}")
    print(f"Финальных прогнозов: {len(forecasts)}")
    print(f"Подборов XGBoost: {len(params)}")


if __name__ == "__main__":
    main()