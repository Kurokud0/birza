import pandas as pd
import cbrapi as cbr


def load_data_cbr(currency_code, date_from, date_till):

    data = cbr.get_time_series(
        currency_code,
        date_from,
        date_till,
        period="D"
    )

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()

    df.columns = [
        "date",
        "value"
    ]

    df["date"] = df["date"].dt.to_timestamp()

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df["currency"] = currency_code

    df["currency_id"] = cbr.get_currency_code(
        currency_code
    )

    df = df[
        [
            "date",
            "currency",
            "currency_id",
            "value"
        ]
    ]

    return df


def load_data_metals(date_from, date_till):

    data = cbr.get_metals_prices(
        date_from,
        date_till,
        period="D"
    )

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

    df = df.rename(
        columns={
            "date": "date"
        }
    )

    df["date"] = df["date"].dt.to_timestamp()

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

    metal_codes = {
        "GOLD": 1,
        "SILVER": 2,
        "PLATINUM": 3,
        "PALLADIUM": 4
    }

    df["metal_code"] = df["metal"].map(
        metal_codes
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = df[
        [
            "date",
            "metal",
            "metal_code",
            "price"
        ]
    ]

    return df


def load_data_key_rate(date_from, date_till):

    data = cbr.get_key_rate(
        date_from,
        date_till,
        period="D"
    )

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.reset_index()

    df.columns = [
        "date",
        "key_rate"
    ]

    df["date"] = df["date"].dt.to_timestamp()

    df["key_rate"] = pd.to_numeric(
        df["key_rate"],
        errors="coerce"
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    date_from = "2020-01-01"
    date_till = "2026-09-01"


    currencies = [
        "USD",
        "EUR",
        "CNY",
        "GBP",
        "JPY",
        "CHF",
        "TRY",
        "AED",
        "KZT",
        "INR"
    ]

    data = []

    for currency_code in currencies:

        print(
            f"Загружаем {currency_code}..."
        )

        df_currency = load_data_cbr(
            currency_code,
            date_from,
            date_till
        )

        if df_currency.empty:

            print(
                f"{currency_code}: данных нет"
            )

            continue

        data.append(df_currency)

        print(
            f"{currency_code}: "
            f"{len(df_currency)} строк"
        )


    df = pd.concat(
        data,
        ignore_index=True
    )


    print("Валюты")

    print("\nИтог:")
    print(df)

    print("\nРазмер:")
    print(df.shape)

    print("\nКоличество записей по валютам:")
    print(
        df.groupby("currency").size()
    )

    print("\nТипы данных:")
    print(df.dtypes)


    print("Драгоценные металлы")
    print("\nЗагружаем драгоценные металлы...")

    metals = load_data_metals(
        date_from,
        date_till
    )


    if metals.empty:

        print("Данных по металлам нет")

    else:

        print("\nДрагоценные металлы:")
        print(metals)

        print("\nРазмер:")
        print(metals.shape)

        print("\nКоличество записей по металлам:")
        print(
            metals.groupby("metal").size()
        )

        print("\nТипы данных:")
        print(metals.dtypes)


    print("Ключевая ставка")
    print("\nЗагружаем ключевую ставку...")

    key_rate = load_data_key_rate(
        date_from,
        date_till
    )


    if key_rate.empty:

        print("Данных по ключевой ставке нет")

    else:

        print("\nКлючевая ставка:")
        print(key_rate)

        print("\nРазмер:")
        print(key_rate.shape)

        print("\nТипы данных:")
        print(key_rate.dtypes)