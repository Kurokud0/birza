from datetime import date, timedelta, datetime
from time import sleep

from db import get_connection

from moex import (
    load_data_moex,
    save_to_raw,
    load_data_index,
    save_index_to_raw
)

from cbr import (
    load_data_cbr,
    load_data_metals,
    load_data_key_rate,
    load_data_zcyc,
    save_currency_to_raw,
    save_metals_to_raw,
    save_key_rate_to_raw,
    save_zcyc_to_raw
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


INDICES = [
    "IMOEX"
]


DEFAULT_START = date(2020, 1, 1)
RETRY_ATTEMPTS = 3
RETRY_DELAYS = [2, 5, 10]


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

    try:
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

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def retry_call(func, *args, **kwargs):
    last_error = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            return func(*args, **kwargs)

        except Exception as error:
            last_error = error

            if attempt == RETRY_ATTEMPTS - 1:
                break

            delay = RETRY_DELAYS[
                min(attempt, len(RETRY_DELAYS) - 1)
            ]

            print(
                f"Ошибка временного характера: {error}"
            )

            print(
                f"Повторная попытка "
                f"{attempt + 2}/{RETRY_ATTEMPTS} "
                f"через {delay} сек."
            )

            sleep(delay)

    raise last_error


def run_moex():
    for ticker in TICKERS:
        start_time = datetime.now()
        load_type = "incremental"

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

            if date_from > date.today():
                write_log(
                    "MOEX ISS",
                    "raw.moex_candles",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )

                print(
                    f"{ticker}: новых данных нет"
                )

                continue

            date_till = date.today()

            print(
                f"\nMOEX {ticker}: "
                f"{date_from} → {date_till}"
            )

            df = retry_call(
                load_data_moex,
                ticker,
                str(date_from),
                str(date_till)
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

                print(
                    f"{ticker}: новых данных нет"
                )

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
            try:
                write_log(
                    "MOEX ISS",
                    "raw.moex_candles",
                    load_type,
                    start_time,
                    datetime.now(),
                    "FAILED",
                    0,
                    str(error)
                )
            except Exception as log_error:
                print(
                    f"{ticker}: ошибка журнала — "
                    f"{log_error}"
                )

            print(
                f"{ticker}: ошибка — {error}"
            )


def run_indices():
    for index_code in INDICES:
        start_time = datetime.now()
        load_type = "incremental"

        try:
            last_date = get_last_date(
                "raw.moex_indices",
                "trading_date",
                f"index_code = '{index_code}'"
            )

            if last_date is None:
                date_from = DEFAULT_START
                load_type = "full"
            else:
                date_from = last_date + timedelta(days=1)

            if date_from > date.today():
                write_log(
                    "MOEX ISS",
                    "raw.moex_indices",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )

                print(
                    f"{index_code}: новых данных нет"
                )

                continue

            date_till = date.today()

            print(
                f"\nMOEX {index_code}: "
                f"{date_from} → {date_till}"
            )

            df = retry_call(
                load_data_index,
                index_code,
                str(date_from),
                str(date_till)
            )

            if df.empty:
                write_log(
                    "MOEX ISS",
                    "raw.moex_indices",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )

                print(
                    f"{index_code}: новых данных нет"
                )

                continue

            save_index_to_raw(df)

            rows_loaded = len(df)

            write_log(
                "MOEX ISS",
                "raw.moex_indices",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                rows_loaded
            )

            print(
                f"{index_code}: успешно, "
                f"{rows_loaded} строк"
            )

        except Exception as error:
            try:
                write_log(
                    "MOEX ISS",
                    "raw.moex_indices",
                    load_type,
                    start_time,
                    datetime.now(),
                    "FAILED",
                    0,
                    str(error)
                )
            except Exception as log_error:
                print(
                    f"{index_code}: ошибка журнала — "
                    f"{log_error}"
                )

            print(
                f"{index_code}: ошибка — {error}"
            )


def run_currency():
    for currency in CURRENCIES:
        start_time = datetime.now()
        load_type = "incremental"

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

            if date_from > date.today():
                write_log(
                    "CBR",
                    "raw.cbr_currency",
                    load_type,
                    start_time,
                    datetime.now(),
                    "SUCCESS",
                    0
                )

                print(
                    f"{currency}: новых данных нет"
                )

                continue

            date_till = date.today()

            print(
                f"\nCBR {currency}: "
                f"{date_from} → {date_till}"
            )

            df = retry_call(
                load_data_cbr,
                currency,
                str(date_from),
                str(date_till)
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

                print(
                    f"{currency}: новых данных нет"
                )

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
            try:
                write_log(
                    "CBR",
                    "raw.cbr_currency",
                    load_type,
                    start_time,
                    datetime.now(),
                    "FAILED",
                    0,
                    str(error)
                )
            except Exception as log_error:
                print(
                    f"{currency}: ошибка журнала — "
                    f"{log_error}"
                )

            print(
                f"{currency}: ошибка — {error}"
            )


def run_metals():
    start_time = datetime.now()
    load_type = "incremental"

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

        if date_from > date.today():
            write_log(
                "CBR",
                "raw.cbr_metals",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )

            print(
                "Металлы: новых данных нет"
            )

            return

        date_till = date.today()

        print(
            f"\nCBR металлы: "
            f"{date_from} → {date_till}"
        )

        df = retry_call(
            load_data_metals,
            str(date_from),
            str(date_till)
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

            print(
                "Металлы: новых данных нет"
            )

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
        try:
            write_log(
                "CBR",
                "raw.cbr_metals",
                load_type,
                start_time,
                datetime.now(),
                "FAILED",
                0,
                str(error)
            )
        except Exception as log_error:
            print(
                f"Металлы: ошибка журнала — "
                f"{log_error}"
            )

        print(
            f"Металлы: ошибка — {error}"
        )


def run_key_rate():
    start_time = datetime.now()
    load_type = "incremental"

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

        if date_from > date.today():
            write_log(
                "CBR",
                "raw.cbr_key_rate",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )

            print(
                "Ключевая ставка: новых данных нет"
            )

            return

        date_till = date.today()

        print(
            f"\nCBR ключевая ставка: "
            f"{date_from} → {date_till}"
        )

        df = retry_call(
            load_data_key_rate,
            str(date_from),
            str(date_till)
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

            print(
                "Ключевая ставка: новых данных нет"
            )

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
        try:
            write_log(
                "CBR",
                "raw.cbr_key_rate",
                load_type,
                start_time,
                datetime.now(),
                "FAILED",
                0,
                str(error)
            )
        except Exception as log_error:
            print(
                f"Ключевая ставка: ошибка журнала — "
                f"{log_error}"
            )

        print(
            f"Ключевая ставка: ошибка — {error}"
        )


def run_zcyc():
    start_time = datetime.now()
    load_type = "incremental"

    try:
        last_date = get_last_date(
            "raw.cbr_zcyc",
            "date"
        )

        if last_date is None:
            date_from = DEFAULT_START
            load_type = "full"
        else:
            date_from = last_date + timedelta(days=1)

        if date_from > date.today():
            write_log(
                "CBR",
                "raw.cbr_zcyc",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )

            print(
                "ZCYC: новых данных нет"
            )

            return

        date_till = date.today()

        print(
            f"\nCBR ZCYC: "
            f"{date_from} → {date_till}"
        )

        df = retry_call(
            load_data_zcyc,
            str(date_from),
            str(date_till)
        )

        if df.empty:
            write_log(
                "CBR",
                "raw.cbr_zcyc",
                load_type,
                start_time,
                datetime.now(),
                "SUCCESS",
                0
            )

            print(
                "ZCYC: новых данных нет"
            )

            return

        save_zcyc_to_raw(df)

        rows_loaded = len(df)

        write_log(
            "CBR",
            "raw.cbr_zcyc",
            load_type,
            start_time,
            datetime.now(),
            "SUCCESS",
            rows_loaded
        )

        print(
            f"ZCYC: успешно, "
            f"{rows_loaded} строк"
        )

    except Exception as error:
        try:
            write_log(
                "CBR",
                "raw.cbr_zcyc",
                load_type,
                start_time,
                datetime.now(),
                "FAILED",
                0,
                str(error)
            )
        except Exception as log_error:
            print(
                f"ZCYC: ошибка журнала — "
                f"{log_error}"
            )

        print(
            f"ZCYC: ошибка — {error}"
        )


def main():
    print("ЗАПУСК ETL")

    print("\n[1/6] MOEX АКЦИИ")
    run_moex()

    print("\n[2/6] MOEX ИНДЕКСЫ")
    run_indices()

    print("\n[3/6] CBR ВАЛЮТЫ")
    run_currency()

    print("\n[4/6] CBR МЕТАЛЛЫ")
    run_metals()

    print("\n[5/6] CBR КЛЮЧЕВАЯ СТАВКА")
    run_key_rate()

    print("\n[6/6] CBR ZCYC")
    run_zcyc()

    print("ETL ЗАВЕРШЕН")


if __name__ == "__main__":
    main()