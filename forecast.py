import pandas as pd
from db import get_connection


QUERY = """
SELECT
    trading_date,
    close,
    volume,
    usd_rate,
    eur_rate,
    gold_price,
    key_rate,
    close_lag_1,
    close_lag_3,
    close_lag_5,
    return_1d,
    ma_5,
    ma_20,
    target_close
FROM mart.stock_features
WHERE target_close IS NOT NULL
ORDER BY trading_date;
"""


def load_features():
    connection = get_connection()

    df = pd.read_sql(
        QUERY,
        connection
    )

    connection.close()

    return df


if __name__ == "__main__":

    df = load_features()

    print("Размер датасета:")
    print(df.shape)

    print("\nПервые строки:")
    print(df.head())

    print("\nПропуски:")
    print(df.isnull().sum())