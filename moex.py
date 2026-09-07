import requests
import pandas as pd


def load_data_moex(ticker, date_from, date_till):

    moex_url = (
        f"https://iss.moex.com/iss/engines/stock/"
        f"markets/shares/securities/{ticker}/candles.json"
    )

    full_data = []
    iter_start = 0

    while True:

        params = {
            "from": date_from,
            "till": date_till,
            "interval": 24,
            "start": iter_start,
        }

        response = requests.get(
            moex_url,
            params=params
        )

        response.raise_for_status()

        candles = response.json()["candles"]
        rows = candles["data"]

        if not rows:
            break

        df_part = pd.DataFrame(
            rows,
            columns=candles["columns"]
        )

        full_data.append(df_part)

        print(
            f"{ticker}: загружено "
            f"{sum(len(x) for x in full_data)} строк"
        )

        iter_start += len(rows)

    if not full_data:
        return pd.DataFrame()

    df = pd.concat(
        full_data,
        ignore_index=True
    )

    df["ticker"] = ticker

    df["begin"] = pd.to_datetime(
        df["begin"],
        errors="coerce"
    )

    df["end"] = pd.to_datetime(
        df["end"],
        errors="coerce"
    )

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "value",
        "volume"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.drop_duplicates(
        subset=[
            "ticker",
            "begin"
        ]
    )

    df = df.sort_values(
        [
            "ticker",
            "begin"
        ]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    tickers = [
        "SBER",
        "GAZP",
        "LKOH",
        "GMKN",
        "YDEX",
        "ROSN",
        "NVTK",
        "TATN",
        "MGNT",
        "MTSS"
    ]

    date_from = "2020-01-01"
    date_till = "2026-09-01"

    data = []

    for ticker in tickers:

        print(
            f"\nЗагружаем {ticker}..."
        )

        df_ticker = load_data_moex(
            ticker,
            date_from,
            date_till
        )

        if df_ticker.empty:

            print(
                f"{ticker}: данных нет"
            )

            continue

        data.append(df_ticker)

        print(
            f"{ticker}: "
            f"{len(df_ticker)} строк"
        )

    if data:

        df = pd.concat(
            data,
            ignore_index=True
        )

        print("\nИтог:")
        print(df)

        print("\nРазмер:")
        print(df.shape)

        print("\nКоличество записей по компаниям:")
        print(
            df.groupby("ticker").size()
        )

        print("\nПропуски:")
        print(
            df.isna().sum()
        )

        print("\nТипы данных:")
        print(df.dtypes)



from db import get_connection


def save_to_raw(df):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO raw.moex_candles (
            open,
            close,
            high,
            low,
            value,
            volume,
            begin,
            end_time,
            ticker
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, begin)
        DO UPDATE SET
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            value = EXCLUDED.value,
            volume = EXCLUDED.volume,
            end_time = EXCLUDED.end_time;
    """

    for _, row in df.iterrows():
        cursor.execute(
            query,
            (
                row["open"],
                row["close"],
                row["high"],
                row["low"],
                row["value"],
                row["volume"],
                row["begin"],
                row["end"],
                row["ticker"],
            )
        )

    connection.commit()

    cursor.close()
    connection.close()

    print(f"Загружено строк: {len(df)}")