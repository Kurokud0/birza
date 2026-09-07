import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from db import get_connection


N_SPLITS = 5
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


def load_data():
    conn = get_connection()

    query = """
        SELECT
            trading_date,
            close,
            volume,
            usd_rate,
            gold_price
        FROM mart.market_daily
        ORDER BY trading_date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["trading_date"] = pd.to_datetime(df["trading_date"])

    for column in ["close", "volume", "usd_rate", "gold_price"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("trading_date").reset_index(drop=True)


def create_features(df):
    df = df.copy()
    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)
    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    ma_5 = df["close"].rolling(5).mean()
    ma_20 = df["close"].rolling(20).mean()
    df["ma_ratio_5"] = df["close"] / ma_5 - 1
    df["ma_ratio_20"] = df["close"] / ma_20 - 1
    future_return = df["close"].shift(-1) / df["close"] - 1
    df["target"] = (future_return > 0).astype(int)

    return df


def create_models():
    return {
        "Dummy": DummyClassifier(
            strategy="most_frequent"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            random_state=RANDOM_STATE
        ),
    }


def calculate_metrics(y_true, y_pred):
    return {
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


def evaluate_time_series(X, y, model_name, model):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []
    print(f"МОДЕЛЬ: {model_name}")
    for fold, (train_index, test_index) in enumerate(
        tscv.split(X),
        start=1
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(
            y_test,
            predictions
        )
        fold_results.append({
            "fold": fold,
            **metrics
        })

        print(f"\nФолд {fold}")
        print(f"Train: {len(train_index)}")
        print(f"Test:  {len(test_index)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")

    fold_df = pd.DataFrame(fold_results)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]

    mean_metrics = fold_df[metric_columns].mean()
    std_metrics = fold_df[metric_columns].std()

    print("СРЕДНИЕ ЗНАЧЕНИЯ")

    for metric in metric_columns:
        print(
            f"{metric.replace('_', ' ').title()}: "
            f"{mean_metrics[metric]:.4f} "
            f"+/- "
            f"{std_metrics[metric]:.4f}"
        )

    return fold_df, mean_metrics, std_metrics


def evaluate_holdout(X, y, model_name, model):
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = calculate_metrics(
        y_test,
        predictions
    )

    print(f"HOLDOUT: {model_name}")
    print(f"Train: {len(X_train)}")
    print(f"Test:  {len(X_test)}")

    for metric, value in metrics.items():
        print(
            f"{metric.replace('_', ' ').title()}: "
            f"{value:.4f}"
        )

    return metrics


def make_latest_prediction(df, model):
    data = df.dropna(subset=FEATURES).copy()
    X = data[FEATURES]
    y = data["target"]
    model.fit(X, y)
    latest_X = X.iloc[[-1]]
    prediction = int(model.predict(latest_X)[0])
    probability_growth = np.nan

    if hasattr(model, "predict_proba"):
        probability_growth = float(
            model.predict_proba(latest_X)[0][1]
        )

    return (
        data["trading_date"].iloc[-1],
        data["close"].iloc[-1],
        prediction,
        probability_growth
    )


def main():
    print("CLASSIFICATION — STABILITY CHECK")
    df = load_data()
    print(f"\nИсходные данные: {df.shape}")
    print(
        f"Период: "
        f"{df['trading_date'].min().date()} "
        f"— "
        f"{df['trading_date'].max().date()}"
    )

    df = create_features(df)

    data = df[
        ["trading_date", "close"] +
        FEATURES +
        ["target"]
    ].dropna()

    X = data[FEATURES]
    y = data["target"]

    print(f"\nПосле подготовки: {data.shape}")
    print(f"Количество признаков: {len(FEATURES)}")
    print(f"Размер выборки: {X.shape}")

    class_distribution = (
        y.value_counts(normalize=True)
        .sort_index()
    )

    print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
    print(
        f"DOWN: "
        f"{class_distribution.get(0, 0) * 100:.2f}%"
    )

    print(
        f"GROWTH: "
        f"{class_distribution.get(1, 0) * 100:.2f}%"
    )

    models = create_models()
    results = {}
    for model_name, model in models.items():
        fold_df, mean_metrics, std_metrics = (
            evaluate_time_series(
                X,
                y,
                model_name,
                model
            )
        )

        results[model_name] = {
            "folds": fold_df,
            "mean": mean_metrics,
            "std": std_metrics,
        }

    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    comparison = []
    for model_name, result in results.items():
        comparison.append({
            "model": model_name,
            "accuracy": result["mean"]["accuracy"],
            "balanced_accuracy": result["mean"]["balanced_accuracy"],
            "balanced_std": result["std"]["balanced_accuracy"],
            "f1": result["mean"]["f1"],
        })

    comparison_df = (
        pd.DataFrame(comparison)
        .sort_values(
            "balanced_accuracy",
            ascending=False
        )
    )

    print(
        comparison_df.to_string(index=False)
    )

    best_model_name = comparison_df.iloc[0]["model"]

    print(
        f"\nЛучшая модель по Balanced Accuracy: "
        f"{best_model_name}"
    )

    holdout_model = create_models()[best_model_name]

    holdout_metrics = evaluate_holdout(
        X,
        y,
        best_model_name,
        holdout_model
    )

    final_model = create_models()[best_model_name]

    (
        latest_date,
        latest_close,
        prediction,
        probability_growth
    ) = make_latest_prediction(
        df,
        final_model
    )

    print("ФИНАЛЬНЫЙ ПРОГНОЗ")
    print(f"\nДата: {latest_date.date()}")
    print(f"Цена закрытия: {latest_close:.2f}")
    print("Горизонт: 1 торговый день")
    print(f"Модель: {best_model_name}")
    print(
        f"Прогноз: "
        f"{'GROWTH' if prediction == 1 else 'DOWN'}"
    )

    if not np.isnan(probability_growth):
        print(
            f"Вероятность роста: "
            f"{probability_growth * 100:.2f}%"
        )

        print(
            f"Вероятность снижения: "
            f"{(1 - probability_growth) * 100:.2f}%"
        )

    print("ИТОГ")
    best_result = results[best_model_name]
    print(f"\nМодель: {best_model_name}")
    print(
        f"Средняя Balanced Accuracy: "
        f"{best_result['mean']['balanced_accuracy']:.4f}"
    )
    print(
        f"Стандартное отклонение: "
        f"{best_result['std']['balanced_accuracy']:.4f}"
    )
    print(
        f"Holdout Balanced Accuracy: "
        f"{holdout_metrics['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()