import io
import time

import cbrapi as cbr
import pandas as pd
import requests

from db import get_connection


REQUEST_TIMEOUT = 30
RETRIES = 3
RETRY_DELAYS = [2, 5, 10]

ZCYC_MATURITIES = [
    0.25, 0.50, 0.75, 1.00, 2.00, 3.00,
    5.00, 7.00, 10.00, 15.00, 20.00, 30.00
]

METAL_CODES = {
    "GOLD": 1,
    "SILVER": 2,
    "PLATINUM": 3,
    "PALLADIUM": 4
}


def retry(func, *args, **kwargs):
    last_error = None

    for attempt in range(RETRIES):
        try:
            return func(*args, **kwargs)

        except Exception as error:
            last_error = error

            if attempt == RETRIES - 1:
                break

            delay = RETRY_DELAYS[attempt]

            print(
                f"Ошибка: {error}. "
                f"Повтор через {delay} сек."
            )

            time.sleep(delay)

    raise last_error


def clean_dates(df, column="date"):
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    return df


def load_data_cbr(currency_code, date_from, date_till):
    data = retry(
        cbr.get_time_series,
        currency_code,
        date_from,
        date_till,
        period="D"
    )

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()
    df.columns = ["date", "value"]

    df = clean_dates(df)
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df["currency"] = currency_code
    df["currency_id"] = (
        cbr.get_currency_code(currency_code).strip()
    )

    df = df.dropna(
        subset=["date", "value"]
    )

    df = df.drop_duplicates(
        subset=["date", "currency"]
    )

    return df[
        ["date", "currency", "currency_id", "value"]
    ].sort_values("date").reset_index(drop=True)


def load_data_metals(date_from, date_till):
    data = retry(
        cbr.get_metals_prices,
        date_from,
        date_till,
        period="D"
    )

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    df = clean_dates(df)

    df = df[
        [
            "date",
            "gold",
            "silver",
            "platinum",
            "palladium"
        ]
    ]

    df = df.melt(
        id_vars="date",
        var_name="metal",
        value_name="price"
    )

    df["metal"] = df["metal"].str.upper()
    df["metal_code"] = df["metal"].map(METAL_CODES)
    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "price", "metal_code"]
    )

    df = df.drop_duplicates(
        subset=["date", "metal"]
    )

    return df[
        ["date", "metal", "metal_code", "price"]
    ].sort_values(
        ["date", "metal"]
    ).reset_index(drop=True)


def load_data_key_rate(date_from, date_till):
    data = retry(
        cbr.get_key_rate,
        date_from,
        date_till,
        period="D"
    )

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()
    df.columns = ["date", "key_rate"]

    df = clean_dates(df)

    df["key_rate"] = pd.to_numeric(
        df["key_rate"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "key_rate"]
    )

    return df.drop_duplicates(
        subset=["date"]
    ).sort_values(
        "date"
    ).reset_index(drop=True)


def load_data_zcyc(date_from, date_till):
    url = "https://www.cbr.ru/eng/hd_base/zcyc_params/"

    params = {
        "UniDbQuery.From": pd.to_datetime(
            date_from
        ).strftime("%d.%m.%Y"),
        "UniDbQuery.To": pd.to_datetime(
            date_till
        ).strftime("%d.%m.%Y"),
        "UniDbQuery.Posted": "True"
    }

    response = retry(
        requests.get,
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()
    response.encoding = (
        response.encoding
        or response.apparent_encoding
        or "utf-8"
    )

    try:
        tables = pd.read_html(
            io.StringIO(response.text)
        )
    except Exception as error:
        raise RuntimeError(
            "Не удалось прочитать таблицу ZCYC. "
            "Проверь наличие lxml."
        ) from error

    if not tables:
        raise ValueError(
            "Банк России не вернул таблицу ZCYC."
        )

    table = None

    for candidate in tables:
        text = " ".join(
            map(str, candidate.columns)
        )

        if any(
            x in text
            for x in ["0.25", "0,25", "Date"]
        ):
            table = candidate
            break

    if table is None:
        for candidate in tables:
            if len(candidate.columns) >= 13:
                table = candidate
                break

    if table is None:
        raise ValueError(
            "Не удалось определить таблицу ZCYC."
        )

    table = table.copy()

    if isinstance(
        table.columns,
        pd.MultiIndex
    ):
        table.columns = [
            next(
                (
                    str(part).strip()
                    for part in reversed(column)
                    if str(part).strip()
                    and str(part).lower() != "nan"
                ),
                ""
            )
            for column in table.columns
        ]

    else:
        table.columns = [
            str(column).strip()
            for column in table.columns
        ]

    date_column = next(
        (
            column
            for column in table.columns
            if "date" in str(column).lower()
            or "дата" in str(column).lower()
        ),
        table.columns[0]
    )

    table = table.rename(
        columns={date_column: "date"}
    )

    maturity_columns = {}

    for column in table.columns:
        if column == "date":
            continue

        try:
            maturity = float(
                str(column)
                .strip()
                .replace(",", ".")
            )
        except ValueError:
            continue

        if maturity in ZCYC_MATURITIES:
            maturity_columns[column] = maturity

    if not maturity_columns:
        raise ValueError(
            "Не удалось определить сроки ZCYC."
        )

    table = table[
        ["date"] + list(maturity_columns)
    ].melt(
        id_vars="date",
        var_name="maturity_column",
        value_name="yield_percent"
    )

    table["maturity_years"] = (
        table["maturity_column"]
        .map(maturity_columns)
    )

    table["date"] = pd.to_datetime(
        table["date"],
        errors="coerce",
        dayfirst=True
    ).dt.date

    table["yield_percent"] = (
        table["yield_percent"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
    )

    table["yield_percent"] = pd.to_numeric(
        table["yield_percent"],
        errors="coerce"
    )

    table = table.dropna(
        subset=[
            "date",
            "maturity_years",
            "yield_percent"
        ]
    )

    table = table[
        (table["yield_percent"] >= 0)
        & table["maturity_years"].isin(
            ZCYC_MATURITIES
        )
    ]

    return table[
        [
            "date",
            "maturity_years",
            "yield_percent"
        ]
    ].drop_duplicates(
        subset=["date", "maturity_years"]
    ).sort_values(
        ["date", "maturity_years"]
    ).reset_index(drop=True)


def save_raw(
    df,
    table,
    columns,
    conflict,
    update
):
    if df.empty:
        return

    connection = get_connection()
    cursor = connection.cursor()

    fields = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    query = f"""
        INSERT INTO {table} ({fields})
        VALUES ({placeholders})
        ON CONFLICT {conflict}
        DO UPDATE SET {update},
        loaded_at = CURRENT_TIMESTAMP
    """

    try:
        for _, row in df.iterrows():
            cursor.execute(
                query,
                tuple(row[column] for column in columns)
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def save_currency_to_raw(df):
    save_raw(
        df,
        "raw.cbr_currency",
        ["date", "currency", "currency_id", "value"],
        "(date, currency)",
        "currency_id = EXCLUDED.currency_id, "
        "value = EXCLUDED.value"
    )

    print(f"Валюты: {len(df)} строк")


def save_metals_to_raw(df):
    save_raw(
        df,
        "raw.cbr_metals",
        ["date", "metal", "metal_code", "price"],
        "(date, metal)",
        "metal_id = EXCLUDED.metal_code, "
        "price = EXCLUDED.price"
    )

    print(f"Металлы: {len(df)} строк")


def save_key_rate_to_raw(df):
    save_raw(
        df,
        "raw.cbr_key_rate",
        ["date", "key_rate"],
        "(date)",
        "key_rate = EXCLUDED.key_rate"
    )

    print(f"Ключевая ставка: {len(df)} строк")


def save_zcyc_to_raw(df):
    save_raw(
        df,
        "raw.cbr_zcyc",
        ["date", "maturity_years", "yield_percent"],
        "(date, maturity_years)",
        "yield_percent = EXCLUDED.yield_percent"
    )

    print(f"ZCYC: {len(df)} строк")


if __name__ == "__main__":
    date_from = "2020-01-01"
    date_till = "2026-09-21"

    currencies = [
        "USD", "EUR", "CNY", "GBP", "JPY",
        "CHF", "TRY", "AED", "KZT", "INR"
    ]

    print("CBR: ВАЛЮТЫ")

    for currency in currencies:
        try:
            df = load_data_cbr(
                currency,
                date_from,
                date_till
            )

            print(
                f"{currency}: {len(df)} строк"
            )

        except Exception as error:
            print(
                f"{currency}: ошибка — {error}"
            )

    print("\nCBR: МЕТАЛЛЫ")

    try:
        df = load_data_metals(
            date_from,
            date_till
        )
        print(
            f"Металлы: {len(df)} строк"
        )

    except Exception as error:
        print(
            f"Металлы: ошибка — {error}"
        )

    print("\nCBR: КЛЮЧЕВАЯ СТАВКА")

    try:
        df = load_data_key_rate(
            date_from,
            date_till
        )
        print(
            f"Ключевая ставка: {len(df)} строк"
        )

    except Exception as error:
        print(
            f"Ключевая ставка: ошибка — {error}"
        )

    print("\nCBR: ZCYC")

    try:
        df = load_data_zcyc(
            date_from,
            date_till
        )

        print(
            f"ZCYC: {len(df)} строк"
        )

        if not df.empty:
            print(
                f"Период: {df['date'].min()} "
                f"→ {df['date'].max()}"
            )

            print(
                "Сроки:",
                sorted(
                    df["maturity_years"].unique()
                )
            )

    except Exception as error:
        print(
            f"ZCYC: ошибка — {error}"
        )