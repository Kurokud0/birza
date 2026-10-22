import requests
import pandas as pd

from datetime import date, timedelta

from db import get_connection


REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]


TICKER_ALIASES = {
    "TCSG": ("TCSG", "T", "2024-11-28")
}


def request_json(url, params):
    last_error = None

    for attempt in range(REQUEST_RETRIES):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if 500 <= response.status_code < 600:
                raise requests.HTTPError(
                    f"HTTP {response.status_code}",
                    response=response
                )

            response.raise_for_status()

            return response.json()

        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.HTTPError,
            requests.RequestException,
            ValueError
        ) as error:

            last_error = error

            if (
                isinstance(error, requests.HTTPError)
                and error.response is not None
                and error.response.status_code < 500
            ):
                raise

            if attempt == REQUEST_RETRIES - 1:
                break

            delay = RETRY_DELAYS[
                min(attempt, len(RETRY_DELAYS) - 1)
            ]

            print(
                f"Ошибка запроса MOEX: {error}"
            )

            print(
                f"Повторная попытка "
                f"{attempt + 2}/{REQUEST_RETRIES} "
                f"через {delay} сек."
            )

            import time
            time.sleep(delay)

    raise last_error


def load_moex_part(ticker, date_from, date_till):

    date_from = pd.to_datetime(date_from).date()
    date_till = pd.to_datetime(date_till).date()

    if date_from > date_till:
        return pd.DataFrame()

    url = (
        "https://iss.moex.com/iss/engines/stock/"
        f"markets/shares/securities/{ticker}/candles.json"
    )

    full_data = []
    start = 0

    while True:
        params = {
            "from": date_from.isoformat(),
            "till": date_till.isoformat(),
            "interval": 24,
            "start": start
        }

        data = request_json(
            url,
            params
        )

        candles = data.get("candles")

        if not candles:
            break

        rows = candles.get("data", [])
        columns = candles.get("columns", [])

        if not rows:
            break

        full_data.append(
            pd.DataFrame(
                rows,
                columns=columns
            )
        )

        start += len(rows)

    if not full_data:
        return pd.DataFrame()

    return pd.concat(
        full_data,
        ignore_index=True
    )


def load_data_moex(ticker, date_from, date_till):

    date_from = pd.to_datetime(date_from).date()
    date_till = pd.to_datetime(date_till).date()

    if date_from > date_till:
        return pd.DataFrame()

    if ticker in TICKER_ALIASES:
        old_ticker, new_ticker, switch_date = TICKER_ALIASES[ticker]

        switch_date = pd.to_datetime(
            switch_date
        ).date()

        parts = []

        if date_from < switch_date:
            old_end = min(
                date_till,
                switch_date - timedelta(days=1)
            )

            old_data = load_moex_part(
                old_ticker,
                date_from,
                old_end
            )

            if not old_data.empty:
                old_data["ticker"] = ticker
                parts.append(old_data)

        new_start = max(
            date_from,
            switch_date
        )

        if new_start <= date_till:
            new_data = load_moex_part(
                new_ticker,
                new_start,
                date_till
            )

            if not new_data.empty:
                new_data["ticker"] = ticker
                parts.append(new_data)

        if not parts:
            return pd.DataFrame()

        df = pd.concat(
            parts,
            ignore_index=True
        )

    else:
        df = load_moex_part(
            ticker,
            date_from,
            date_till
        )

        if df.empty:
            return pd.DataFrame()

        df["ticker"] = ticker

    df["begin"] = pd.to_datetime(
        df["begin"],
        errors="coerce"
    )

    if "end" in df.columns:
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

    df = df.dropna(
        subset=[
            "ticker",
            "begin"
        ]
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


def save_to_raw(df):

    if df.empty:
        return

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
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT (ticker, begin)
        DO UPDATE SET
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            value = EXCLUDED.value,
            volume = EXCLUDED.volume,
            end_time = EXCLUDED.end_time,
            loaded_at = CURRENT_TIMESTAMP;
    """

    try:
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
                    row["ticker"]
                )
            )

        connection.commit()

        print(
            f"MOEX: сохранено/обновлено строк: {len(df)}"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def load_data_index(index_code, date_from, date_till):

    date_from = pd.to_datetime(date_from).date()
    date_till = pd.to_datetime(date_till).date()

    if date_from > date_till:
        return pd.DataFrame()

    url = (
        "https://iss.moex.com/iss/engines/stock/"
        f"markets/index/securities/{index_code}/candles.json"
    )

    full_data = []
    start = 0

    while True:
        params = {
            "from": date_from.isoformat(),
            "till": date_till.isoformat(),
            "interval": 24,
            "start": start
        }

        data = request_json(
            url,
            params
        )

        candles = data.get("candles")

        if not candles:
            break

        rows = candles.get("data", [])
        columns = candles.get("columns", [])

        if not rows:
            break

        full_data.append(
            pd.DataFrame(
                rows,
                columns=columns
            )
        )

        start += len(rows)

    if not full_data:
        return pd.DataFrame()

    df = pd.concat(
        full_data,
        ignore_index=True
    )

    df["begin"] = pd.to_datetime(
        df["begin"],
        errors="coerce"
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "begin",
            "close"
        ]
    )

    df = df[
        df["close"] > 0
    ]

    df["trading_date"] = df["begin"].dt.date
    df["index_code"] = index_code

    df = df[
        [
            "index_code",
            "trading_date",
            "close"
        ]
    ]

    df = df.drop_duplicates(
        subset=[
            "index_code",
            "trading_date"
        ]
    )

    df = df.sort_values(
        [
            "index_code",
            "trading_date"
        ]
    ).reset_index(drop=True)

    return df


def save_index_to_raw(df):

    if df.empty:
        return

    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO raw.moex_indices (
            index_code,
            trading_date,
            close
        )
        VALUES (
            %s,
            %s,
            %s
        )
        ON CONFLICT (index_code, trading_date)
        DO UPDATE SET
            close = EXCLUDED.close,
            loaded_at = CURRENT_TIMESTAMP;
    """

    try:
        for _, row in df.iterrows():
            cursor.execute(
                query,
                (
                    row["index_code"],
                    row["trading_date"],
                    row["close"]
                )
            )

        connection.commit()

        print(
            f"MOEX индекс: "
            f"сохранено/обновлено строк: {len(df)}"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


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
        "MTSS",
        "MOEX",
        "VTBR",
        "AFLT",
        "ALRS",
        "CHMF",
        "NLMK",
        "PLZL",
        "PHOR",
        "SNGS",
        "SNGSP",
        "TCSG",
        "OZON",
        "RUAL",
        "PIKK",
        "IRAO"
    ]

    date_from = "2020-01-01"
    date_till = date.today()

    data = []

    for ticker in tickers:
        print(
            f"Загружаем {ticker}..."
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

        data.append(
            df_ticker
        )

        print(
            f"{ticker}: {len(df_ticker)} строк"
        )

    if data:
        df = pd.concat(
            data,
            ignore_index=True
        )

        print("\nИтог:")
        print(
            f"Всего строк: {len(df)}"
        )

        print(
            f"Компаний: {df['ticker'].nunique()}"
        )

        print(
            "\nКоличество записей по компаниям:"
        )

        print(
            df.groupby("ticker").size()
        )

        print(
            "\nПропуски:"
        )

        print(
            df.isna().sum()
        )