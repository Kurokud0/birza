import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier

from db import get_connection


warnings.filterwarnings("ignore")


# Настройки
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


# Загрузка данных
def load_data():
    conn = get_connection()

    query = """
        SELECT
            ticker,
            trading_date,
            close,
            volume,
            usd_rate,
            gold_price
        FROM mart.market_daily
        ORDER BY ticker, trading_date
    """

    df = pd.read_sql(query, conn)

    conn.close()

    df["trading_date"] = pd.to_datetime(df["trading_date"])

    return df


# Признаки
def prepare_features(df):
    df = df.copy()

    df = df.sort_values(["ticker", "trading_date"])

    grouped = df.groupby("ticker", group_keys=False)

    df["return_1d"] = grouped["close"].pct_change(1)
    df["return_3d"] = grouped["close"].pct_change(3)
    df["return_5d"] = grouped["close"].pct_change(5)

    df["volatility_5"] = (
        grouped["close"]
        .pct_change()
        .rolling(5)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["volatility_20"] = (
        grouped["close"]
        .pct_change()
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    ma_5 = (
        grouped["close"]
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ma_20 = (
        grouped["close"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["ma_ratio_5"] = df["close"] / ma_5
    df["ma_ratio_20"] = df["close"] / ma_20

    # Следующий торговый день
    df["next_return"] = (
        grouped["close"].shift(-1) / df["close"] - 1
    )

    # Класс:
    # 1 = рост
    # 0 = снижение
    df["target"] = (df["next_return"] > 0).astype(int)

    df = df.dropna(subset=FEATURES + ["next_return"])

    return df



# Модели
def create_dummy():
    return DummyClassifier(
        strategy="prior"
    )


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


# Пространство XGBOOST
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


# Метрики

def calculate_metrics(y_true, y_pred, y_prob):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )
    else:
        result["roc_auc"] = np.nan

    return result


# Оценка TEST
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities
    )

    return metrics


# Подбор XGBOOST
def tune_xgboost(X_train, y_train):
    tscv = TimeSeriesSplit(
        n_splits=SEARCH_SPLITS
    )

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
    )

    search.fit(X_train, y_train)

    return search.best_params_


# Обучение XGBOOST
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


# Основа
def run_experiment(df):
    results = []
    final_forecasts = []
    best_params = []

    tickers = sorted(df["ticker"].unique())

    total_tasks = len(tickers) * len(PERIODS)
    current_task = 0

    for ticker in tickers:

        ticker_df = df[
            df["ticker"] == ticker
        ].copy()

        for period, start_date in PERIODS.items():

            current_task += 1

            print(
                f"[{current_task}/{total_tasks}] "
                f"{ticker} | {period}"
            )

            period_df = ticker_df[
                ticker_df["trading_date"] >= start_date
            ].copy()

            period_df = period_df.sort_values(
                "trading_date"
            )

            if len(period_df) < 100:
                print("  пропуск: мало данных")
                continue

            X = period_df[FEATURES]
            y = period_df["target"]

            # Разделение на train/test

            split_index = int(
                len(period_df) * (1 - TEST_SIZE)
            )

            X_train = X.iloc[:split_index]
            y_train = y.iloc[:split_index]

            X_test = X.iloc[split_index:]
            y_test = y.iloc[split_index:]

            # DUMMY
            dummy = create_dummy()

            dummy_metrics = evaluate_model(
                dummy,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append({
                "ticker": ticker,
                "period": period,
                "model": "Dummy",
                **dummy_metrics
            })

            # GRADIENT BOOSTING
            gb = create_gradient_boosting()

            gb_metrics = evaluate_model(
                gb,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append({
                "ticker": ticker,
                "period": period,
                "model": "GradientBoosting",
                **gb_metrics
            })

            # XGBOOST HYPERPARAMETER SEARCH

            best_xgb_params = tune_xgboost(
                X_train,
                y_train
            )

            best_params.append({
                "ticker": ticker,
                "period": period,
                **best_xgb_params
            })

            # XGBOOST

            xgb = create_tuned_xgb(
                best_xgb_params
            )

            xgb_metrics = evaluate_model(
                xgb,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append({
                "ticker": ticker,
                "period": period,
                "model": "XGBoost",
                **xgb_metrics
            })

            # ФИНАЛЬНЫЙ XGBOOST
            final_model = create_tuned_xgb(
                best_xgb_params
            )

            final_model.fit(
                X,
                y
            )

            last_row = period_df.iloc[[-1]]

            X_last = last_row[FEATURES]

            prediction = int(
                final_model.predict(X_last)[0]
            )

            probability_growth = float(
                final_model.predict_proba(X_last)[0, 1]
            )

            probability_down = (
                1 - probability_growth
            )

            final_forecasts.append({
                "ticker": ticker,
                "period": period,
                "model": "XGBoost",
                "forecast_date": last_row[
                    "trading_date"
                ].iloc[0],
                "close": last_row[
                    "close"
                ].iloc[0],
                "prediction": (
                    "GROWTH"
                    if prediction == 1
                    else "DOWN"
                ),
                "probability_growth": (
                    f"{probability_growth:.2%}"
                ),
                "probability_down": (
                    f"{probability_down:.2%}"
                ),
            })

    return (
        pd.DataFrame(results),
        pd.DataFrame(final_forecasts),
        pd.DataFrame(best_params),
    )


# СВОДНАЯ СТАТИСТИКА

def print_summary(results):
    print("Средние результаты TEST")

    summary = (
        results
        .groupby(["period", "model"])[
            [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ]
        .mean()
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "accuracy": "{:.2%}".format,
                "balanced_accuracy": "{:.2%}".format,
                "precision": "{:.2%}".format,
                "recall": "{:.2%}".format,
                "f1": "{:.2%}".format,
                "roc_auc": "{:.3f}".format,
            },
        )
    )

    print()
    print("Победители по ACCURACY")

    winners = []

    for period in PERIODS:

        period_results = results[
            results["period"] == period
        ]

        mean_scores = (
            period_results
            .groupby("model")["accuracy"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        winner = mean_scores.index[0]

        winners.append({
            "period": period,
            "winner": winner,
            "accuracy": mean_scores.iloc[0],
        })

    winners_df = pd.DataFrame(winners)

    print(
        winners_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.2%}".format
            }
        )
    )

    print()
    print("=" * 100)
    print("КОЛИЧЕСТВО ПОБЕД XGBOOST")
    print("=" * 100)

    wins = 0
    total = 0

    for _, group in results.groupby(
        ["ticker", "period"]
    ):

        best_model = (
            group
            .sort_values(
                "accuracy",
                ascending=False
            )
            .iloc[0]["model"]
        )

        total += 1

        if best_model == "XGBoost":
            wins += 1

    print(
        f"XGBoost победил в {wins} из {total} "
        f"экспериментов ({wins / total:.2%})"
    )


# ФИНАЛЬНЫЕ ПРОГНОЗЫ

def print_forecasts(forecasts):
    print()
    print("=" * 100)
    print("ФИНАЛЬНЫЕ ПРОГНОЗЫ")
    print("=" * 100)

    print(
        forecasts.to_string(
            index=False
        )
    )


# КОМПАКТНЫЕ ПАРАМЕТРЫ XGBOOST

def print_best_params(params):
    print()
    print("=" * 100)
    print("ЧАСТОТА ВЫБРАННЫХ ПАРАМЕТРОВ XGBOOST")
    print("=" * 100)

    parameter_columns = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]

    for column in parameter_columns:

        counts = (
            params[column]
            .value_counts()
            .head(5)
        )

        print(f"\n{column}:")
        print(counts.to_string())


# MAIN

def main():

    print("=" * 100)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 100)

    df = load_data()

    print(
        f"Загружено строк: {len(df):,}"
    )

    df = prepare_features(df)

    print(
        f"После подготовки признаков: {len(df):,}"
    )

    print()
    print("=" * 100)
    print("ЗАПУСК ЭКСПЕРИМЕНТА")
    print("=" * 100)

    results, forecasts, params = run_experiment(
        df
    )

    # РЕЗУЛЬТАТЫ

    print_summary(results)

    print_forecasts(forecasts)

    print_best_params(params)

    # ИТОГ

    print("ИТОГ")

    print(
        f"Компаний: "
        f"{results['ticker'].nunique()}/25"
    )

    print(
        f"Периодов: "
        f"{results['period'].nunique()}/4"
    )

    print(
        f"Оценок моделей: "
        f"{len(results)}"
    )

    print(
        f"Финальных прогнозов: "
        f"{len(forecasts)}"
    )

    print(
        f"Подборов XGBoost: "
        f"{len(params)}"
    )


if __name__ == "__main__":
    main()