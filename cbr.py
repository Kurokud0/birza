import pandas as pd
import cbrapi as cbr
from db import get_connection


def load_data_cbr(currency_code, date_from, date_till):
    data = cbr.get_time_series(currency_code, date_from, date_till, period="D")

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()
    df.columns = ["date", "value"]
    df["date"] = df["date"].dt.to_timestamp()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["currency"] = currency_code
    df["currency_id"] = cbr.get_currency_code(currency_code).strip()

    return df[["date", "currency", "currency_id", "value"]]


def load_data_metals(date_from, date_till):
    data = cbr.get_metals_prices(date_from, date_till, period="D")

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()
    df.columns = [str(column).lower() for column in df.columns]
    df["date"] = df["date"].dt.to_timestamp()

    df = df[["date", "gold", "silver", "platinum", "palladium"]]
    df = df.melt(id_vars="date", var_name="metal", value_name="price")
    df["metal"] = df["metal"].str.upper()

    metal_codes = {"GOLD": 1, "SILVER": 2, "PLATINUM": 3, "PALLADIUM": 4}
    df["metal_code"] = df["metal"].map(metal_codes)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df[["date", "metal", "metal_code", "price"]]


def load_data_key_rate(date_from, date_till):
    data = cbr.get_key_rate(date_from, date_till, period="D")

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()
    df.columns = ["date", "key_rate"]
    df["date"] = df["date"].dt.to_timestamp()
    df["key_rate"] = pd.to_numeric(df["key_rate"], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


def save_currency_to_raw(df):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO raw.cbr_currency (date, currency, currency_id, value)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (date, currency)
        DO UPDATE SET
            currency_id = EXCLUDED.currency_id,
            value = EXCLUDED.value;
    """

    for _, row in df.iterrows():
        cursor.execute(query, (row["date"], row["currency"], row["currency_id"], row["value"]))

    connection.commit()
    cursor.close()
    connection.close()

    print(f"Валюты: {len(df)} строк")


def save_metals_to_raw(df):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO raw.cbr_metals (date, metal, metal_id, price)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (date, metal)
        DO UPDATE SET
            metal_id = EXCLUDED.metal_id,
            price = EXCLUDED.price;
    """

    for _, row in df.iterrows():
        cursor.execute(query, (row["date"], row["metal"], row["metal_code"], row["price"]))

    connection.commit()
    cursor.close()
    connection.close()

    print(f"Металлы: {len(df)} строк")


def save_key_rate_to_raw(df):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO raw.cbr_key_rate (date, key_rate)
        VALUES (%s, %s)
        ON CONFLICT (date)
        DO UPDATE SET
            key_rate = EXCLUDED.key_rate;
    """

    for _, row in df.iterrows():
        cursor.execute(query, (row["date"], row["key_rate"]))

    connection.commit()
    cursor.close()
    connection.close()

    print(f"Ключевая ставка: {len(df)} строк")


if __name__ == "__main__":
    date_from = "2020-01-01"
    date_till = "2026-09-12"

    currencies = ["USD", "EUR", "CNY", "GBP", "JPY", "CHF", "TRY", "AED", "KZT", "INR"]

    currency_data = []

    for currency_code in currencies:
        df_currency = load_data_cbr(currency_code, date_from, date_till)

        if not df_currency.empty:
            currency_data.append(df_currency)
            print(f"{currency_code}: {len(df_currency)} строк")

    if currency_data:
        currencies_df = pd.concat(currency_data, ignore_index=True)
        print(f"Валюты: {len(currencies_df)} строк")

    metals_df = load_data_metals(date_from, date_till)
    print(f"Металлы: {len(metals_df)} строк")

    key_rate_df = load_data_key_rate(date_from, date_till)
    print(f"Ключевая ставка: {len(key_rate_df)} строк")
