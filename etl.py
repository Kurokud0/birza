from datetime import date, timedelta, datetime
import pandas as pd
from db import get_connection
from moex import load_data_moex, save_to_raw
from cbr import (
    load_data_cbr,
    load_data_metals,
    load_data_key_rate,
    save_currency_to_raw,
    save_metals_to_raw,
    save_key_rate_to_raw
)


TICKERS = [
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

CURRENCIES = [
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

TODAY = date.today()
DEFAULT_START = date(2020, 1, 1)


def get_last_date(table_name, date_column, condition=None):
    connection = get_connection()
    cursor = connection.cursor()

    query = f"""
        SELECT MAX({date_column})
        FROM {table_name}
    """

    if condition:
        query += f" WHERE {condition}"

    cursor.execute(query)
    result = cursor.fetchone()[0]

    cursor.close()
    connection.close()

    if result is None:
        return None

    if isinstance(result, datetime):
        return result.date()

    return result


def write_log(
    source_name,
    table_name,
    load_type,
    start_time,
    end_time,
    status,
    rows_loaded=0,
    error_message=None
):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO etl.load_log (
            source_name,
            table_name,
            load_type,
            start_time,
            end_time,
            status,
            rows_loaded,
            error_message
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            source_name,
            table_name,
            load_type,
            start_time,
            end_time,
            status,
            rows_loaded,
            error_message
        )
    )

    connection.commit()

    cursor.close()
    connection.close()


def run_moex():
    for ticker in TICKERS:
        start_time = datetime.now()

        try:
            last_date = get_last_date(
                "raw.moex_candles",
                "begin",
                f"ticker = '{ticker}'"
            )

            if last_date is None:
                date_from = DEFAULT_START
                load_type = "full"
            else:
                date_from = last_date + timedelta(days=1)
                load_type = "incremental"

            if date_from > TODAY:
                write_log(
                    "MOEX ISS",
                    "raw.moex_candles",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )
                continue

            print(
                f"\nMOEX {ticker}: "
                f"{date_from} → {TODAY}"
            )

            df = load_data_moex(
                ticker,
                str(date_from),
                str(TODAY)
            )

            if df.empty:
                write_log(
                    "MOEX ISS",
                    "raw.moex_candles",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )
                print(f"{ticker}: новых данных нет")
                continue

            save_to_raw(df)

            rows_loaded = len(df)

            write_log(
                "MOEX ISS",
                "raw.moex_candles",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                rows_loaded
            )

            print(
                f"{ticker}: успешно, "
                f"{rows_loaded} строк"
            )

        except Exception as error:
            write_log(
                "MOEX ISS",
                "raw.moex_candles",
                "incremental",
                start_time,
                datetime.now(),
                "FAILED",
                0,
                str(error)
            )

            print(
                f"{ticker}: ошибка — {error}"
            )


def run_currency():
    for currency in CURRENCIES:
        start_time = datetime.now()

        try:
            last_date = get_last_date(
                "raw.cbr_currency",
                "date",
                f"currency = '{currency}'"
            )

            if last_date is None:
                date_from = DEFAULT_START
                load_type = "full"
            else:
                date_from = last_date + timedelta(days=1)
                load_type = "incremental"

            if date_from > TODAY:
                write_log(
                    "CBR",
                    "raw.cbr_currency",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )
                continue

            print(
                f"\nCBR {currency}: "
                f"{date_from} → {TODAY}"
            )

            df = load_data_cbr(
                currency,
                str(date_from),
                str(TODAY)
            )

            if df.empty:
                write_log(
                    "CBR",
                    "raw.cbr_currency",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )
                print(f"{currency}: новых данных нет")
                continue

            save_currency_to_raw(df)

            rows_loaded = len(df)

            write_log(
                "CBR",
                "raw.cbr_currency",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                rows_loaded
            )

            print(
                f"{currency}: успешно, "
                f"{rows_loaded} строк"
            )

        except Exception as error:
            write_log(
                "CBR",
                "raw.cbr_currency",
                "incremental",
                start_time,
                datetime.now(),
                "FAILED",
                0,
                str(error)
            )

            print(
                f"{currency}: ошибка — {error}"
            )


def run_metals():
    start_time = datetime.now()

    try:
        last_date = get_last_date(
            "raw.cbr_metals",
            "date"
        )

        if last_date is None:
            date_from = DEFAULT_START
            load_type = "full"
        else:
            date_from = last_date + timedelta(days=1)
            load_type = "incremental"

        if date_from > TODAY:
            write_log(
                "CBR",
                "raw.cbr_metals",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )
            print("\nМеталлы: новых данных нет")
            return

        print(
            f"\nCBR металлы: "
            f"{date_from} → {TODAY}"
        )

        df = load_data_metals(
            str(date_from),
            str(TODAY)
        )

        if df.empty:
            write_log(
                "CBR",
                "raw.cbr_metals",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )
            print("Металлы: новых данных нет")
            return

        save_metals_to_raw(df)

        rows_loaded = len(df)

        write_log(
            "CBR",
            "raw.cbr_metals",
            load_type,
            start_time,
            datetime.now(),
            "SUCCESS",
            rows_loaded
        )

        print(
            f"Металлы: успешно, "
            f"{rows_loaded} строк"
        )

    except Exception as error:
        write_log(
            "CBR",
            "raw.cbr_metals",
            "incremental",
            start_time,
            datetime.now(),
            "FAILED",
            0,
            str(error)
        )

        print(
            f"Металлы: ошибка — {error}"
        )


def run_key_rate():
    start_time = datetime.now()

    try:
        last_date = get_last_date(
            "raw.cbr_key_rate",
            "date"
        )

        if last_date is None:
            date_from = DEFAULT_START
            load_type = "full"
        else:
            date_from = last_date + timedelta(days=1)
            load_type = "incremental"

        if date_from > TODAY:
            write_log(
                "CBR",
                "raw.cbr_key_rate",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )
            print("\nКлючевая ставка: новых данных нет")
            return

        print(
            f"\nCBR ключевая ставка: "
            f"{date_from} → {TODAY}"
        )

        df = load_data_key_rate(
            str(date_from),
            str(TODAY)
        )

        if df.empty:
            write_log(
                "CBR",
                "raw.cbr_key_rate",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )
            print("Ключевая ставка: новых данных нет")
            return

        save_key_rate_to_raw(df)

        rows_loaded = len(df)

        write_log(
            "CBR",
            "raw.cbr_key_rate",
            load_type,
            start_time,
            datetime.now(),
            "SUCCESS",
            rows_loaded
        )

        print(
            f"Ключевая ставка: успешно, "
            f"{rows_loaded} строк"
        )

    except Exception as error:
        write_log(
            "CBR",
            "raw.cbr_key_rate",
            "incremental",
            start_time,
            datetime.now(),
            "FAILED",
            0,
            str(error)
        )

        print(
            f"Ключевая ставка: ошибка — {error}"
        )


def main():
    print("ЗАПУСК ETL")
    print(f"Дата запуска: {datetime.now()}")

    print("\n=== MOEX ===")
    run_moex()

    print("\n=== CBR CURRENCY ===")
    run_currency()

    print("\n=== CBR METALS ===")
    run_metals()

    print("\n=== CBR KEY RATE ===")
    run_key_rate()

    print("\nETL ЗАВЕРШЕН")


if __name__ == "__main__":
    main()
