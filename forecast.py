import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA

from db import get_connection


FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5",
    "volatility_20",
    "ma_ratio_5",
    "ma_ratio_20",
    "usd_return_1d",
    "gold_return_1d",
    "volume_ratio_5",
]


def load_data():
    conn = get_connection()

    query = """
        SELECT
            trading_date,
            close,
            volume,
            usd_rate,
            eur_rate,
            gold_price,
            key_rate
        FROM mart.market_daily
        ORDER BY trading_date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df.sort_values("trading_date").reset_index(drop=True)


def prepare_data(df):
    df = df.copy()

    df["return_1d"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    df["volatility_5"] = (
        df["return_1d"]
        .rolling(5)
        .std()
    )

    df["volatility_20"] = (
        df["return_1d"]
        .rolling(20)
        .std()
    )

    ma_5 = df["close"].rolling(5).mean()
    ma_20 = df["close"].rolling(20).mean()

    df["ma_ratio_5"] = df["close"] / ma_5 - 1
    df["ma_ratio_20"] = df["close"] / ma_20 - 1

    df["usd_return_1d"] = df["usd_rate"].pct_change()
    df["gold_return_1d"] = df["gold_price"].pct_change()

    df["volume_ratio_5"] = (
        df["volume"]
        / df["volume"].rolling(5).mean()
    )

    df["target_return"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1
    )

    return df


def calculate_metrics(y_true, predictions):
    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    mask = predictions != 0

    if mask.sum():
        direction = (
            np.mean(
                np.sign(y_true[mask])
                == np.sign(predictions[mask])
            )
            * 100
        )
    else:
        direction = np.nan

    return mae, rmse, direction


def main():
    df = load_data()

    print(f"Исходные данные: {df.shape}")

    print(
        f"Период: "
        f"{df['trading_date'].min()} "
        f"— "
        f"{df['trading_date'].max()}"
    )

    df = prepare_data(df)

    df = df.dropna(
        subset=FEATURES + ["target_return"]
    ).reset_index(drop=True)

    print(f"\nПосле подготовки: {df.shape}")
    print(f"Количество признаков: {len(FEATURES)}")

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    print("\nTRAIN / TEST")
    print(f"TRAIN: {train.shape}")
    print(f"TEST:  {test.shape}")

    print(
        f"TRAIN: "
        f"{train['trading_date'].iloc[0]} "
        f"— "
        f"{train['trading_date'].iloc[-1]}"
    )

    print(
        f"TEST:  "
        f"{test['trading_date'].iloc[0]} "
        f"— "
        f"{test['trading_date'].iloc[-1]}"
    )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train = train["target_return"]
    y_test = test["target_return"]

    naive_predictions = np.zeros(len(y_test))

    naive_mae, naive_rmse, _ = calculate_metrics(
        y_test.to_numpy(),
        naive_predictions
    )

    catboost = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42
    )

    print("\nОбучение CatBoost...")

    catboost.fit(
        X_train,
        y_train
    )

    catboost_predictions = catboost.predict(X_test)

    (
        catboost_mae,
        catboost_rmse,
        catboost_direction
    ) = calculate_metrics(
        y_test.to_numpy(),
        catboost_predictions
    )

    print("\nОбучение ARIMA...")

    arima = ARIMA(
        y_train,
        order=(1, 0, 1)
    ).fit()

    arima_predictions = (
        arima
        .forecast(steps=len(y_test))
        .to_numpy()
    )

    (
        arima_mae,
        arima_rmse,
        arima_direction
    ) = calculate_metrics(
        y_test.to_numpy(),
        arima_predictions
    )

    results = pd.DataFrame({
        "model": [
            "Naive",
            "ARIMA",
            "CatBoost"
        ],
        "MAE": [
            naive_mae,
            arima_mae,
            catboost_mae
        ],
        "RMSE": [
            naive_rmse,
            arima_rmse,
            catboost_rmse
        ],
        "Directional Accuracy": [
            np.nan,
            arima_direction,
            catboost_direction
        ]
    })

    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")

    print(
        results.to_string(
            index=False
        )
    )

    actual_price = (
        test["close"].to_numpy()
        * (1 + y_test.to_numpy())
    )

    naive_price = (
        test["close"].to_numpy()
        * (1 + naive_predictions)
    )

    arima_price = (
        test["close"].to_numpy()
        * (1 + arima_predictions)
    )

    catboost_price = (
        test["close"].to_numpy()
        * (1 + catboost_predictions)
    )

    price_results = {
        "Naive": mean_absolute_error(
            actual_price,
            naive_price
        ),
        "ARIMA": mean_absolute_error(
            actual_price,
            arima_price
        ),
        "CatBoost": mean_absolute_error(
            actual_price,
            catboost_price
        )
    }

    print("\nОШИБКА ПРОГНОЗА ЦЕНЫ")

    for model_name, mae in price_results.items():
        print(
            f"{model_name}: {mae:.4f} ₽"
        )

    last_date = df["trading_date"].iloc[-1]
    last_close = df["close"].iloc[-1]

    X_last = df[FEATURES].iloc[[-1]]

    final_catboost = catboost.predict(X_last)[0]

    final_arima = (
        ARIMA(
            df["target_return"],
            order=(1, 0, 1)
        )
        .fit()
        .forecast(steps=1)
        .iloc[0]
    )

    final_catboost_price = (
        last_close
        * (1 + final_catboost)
    )

    final_arima_price = (
        last_close
        * (1 + final_arima)
    )

    print("\nФИНАЛЬНЫЙ ПРОГНОЗ")

    print(f"Дата: {last_date}")
    print(f"Цена закрытия: {last_close:.2f} ₽")

    print(
        f"ARIMA: "
        f"{final_arima * 100:.3f}% "
        f"→ {final_arima_price:.2f} ₽"
    )

    print(
        f"CatBoost: "
        f"{final_catboost * 100:.3f}% "
        f"→ {final_catboost_price:.2f} ₽"
    )


if __name__ == "__main__":
    main()
