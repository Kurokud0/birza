import requests
import pandas as pd
from db import get_connection


TICKER_ALIASES = {
    "TCSG": ("TCSG", "T", "2024-11-28")
}


def load_moex_part(ticker, date_from, date_till):
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"
    full_data = []
    start = 0

    while True:
        params = {"from": date_from, "till": date_till, "interval": 24, "start": start}
        response = requests.get(url, params=params)
        response.raise_for_status()

        candles = response.json()["candles"]
        rows = candles["data"]

        if not rows:
            break

        full_data.append(pd.DataFrame(rows, columns=candles["columns"]))
        start += len(rows)

    if not full_data:
        return pd.DataFrame()

    return pd.concat(full_data, ignore_index=True)


def load_data_moex(ticker, date_from, date_till):
    if ticker in TICKER_ALIASES:
        old_ticker, new_ticker, switch_date = TICKER_ALIASES[ticker]

        old_data = load_moex_part(old_ticker, date_from, "2024-11-27")
        new_data = load_moex_part(new_ticker, switch_date, date_till)

        parts = []

        if not old_data.empty:
            old_data["ticker"] = ticker
            parts.append(old_data)

        if not new_data.empty:
            new_data["ticker"] = ticker
            parts.append(new_data)

        if not parts:
            return pd.DataFrame()

        df = pd.concat(parts, ignore_index=True)
    else:
        df = load_moex_part(ticker, date_from, date_till)

        if df.empty:
            return pd.DataFrame()

        df["ticker"] = ticker

    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    numeric_columns = ["open", "close", "high", "low", "value", "volume"]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates(subset=["ticker", "begin"])
    df = df.sort_values(["ticker", "begin"]).reset_index(drop=True)

    return df


def save_to_raw(df):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO raw.moex_candles (
            open, close, high, low, value, volume,
            begin, end_time, ticker
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
        cursor.execute(query, (
            row["open"],
            row["close"],
            row["high"],
            row["low"],
            row["value"],
            row["volume"],
            row["begin"],
            row["end"],
            row["ticker"]
        ))

    connection.commit()
    cursor.close()
    connection.close()

    print(f"Загружено строк: {len(df)}")


if __name__ == "__main__":
    tickers = [
        "SBER", "GAZP", "LKOH", "GMKN", "YDEX",
        "ROSN", "NVTK", "TATN", "MGNT", "MTSS",
        "MOEX", "VTBR", "AFLT", "ALRS", "CHMF",
        "NLMK", "PLZL", "PHOR", "SNGS", "SNGSP",
        "TCSG", "OZON", "RUAL", "PIKK", "IRAO"
    ]

    date_from = "2020-01-01"
    date_till = "2026-09-12"

    data = []

    for ticker in tickers:
        print(f"Загружаем {ticker}...")

        df_ticker = load_data_moex(ticker, date_from, date_till)

        if df_ticker.empty:
            print(f"{ticker}: данных нет")
            continue

        data.append(df_ticker)
        print(f"{ticker}: {len(df_ticker)} строк")

    if data:
        df = pd.concat(data, ignore_index=True)

        print("Итог:")
        print(f"Всего строк: {len(df)}")
        print(f"Компаний: {df['ticker'].nunique()}")

        print("Количество записей по компаниям:")
        print(df.groupby("ticker").size())

        print("Пропуски:")
        print(df.isna().sum())