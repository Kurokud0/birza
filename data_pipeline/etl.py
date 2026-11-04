from datetime import date, timedelta, datetime
from portfolio_app.cli import RussianArgumentParser
from time import sleep
from psycopg2 import sql

from portfolio_app.db import get_connection

from data_pipeline.moex import (
    load_data_moex,
    save_to_raw,
    load_data_index,
    save_index_to_raw,
)

from data_pipeline.cbr import (
    load_data_cbr,
    load_data_metals,
    load_data_key_rate,
    load_data_zcyc,
    save_currency_to_raw,
    save_metals_to_raw,
    save_key_rate_to_raw,
    save_zcyc_to_raw,
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
    "IRAO",
]


CURRENCIES = ["USD", "EUR", "CNY", "GBP", "JPY", "CHF", "TRY", "AED", "KZT", "INR"]


INDICES = ["IMOEX"]


DEFAULT_START = date(2020, 1, 1)
RETRY_ATTEMPTS = 3
RETRY_DELAYS = [2, 5, 10]
FULL_RELOAD = False


def get_last_date(table_name, date_column, filter_column=None, filter_value=None):
    """Находит последнюю сохранённую дату; значения фильтра передаются отдельно от SQL."""
    if FULL_RELOAD:
        return None
    query = sql.SQL("SELECT MAX({date}) FROM {table}").format(
        date=sql.Identifier(date_column), table=sql.Identifier(*table_name.split("."))
    )
    parameters = ()
    if filter_column is not None:
        query += sql.SQL(" WHERE {} = %s").format(sql.Identifier(filter_column))
        parameters = (filter_value,)
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            result = cursor.fetchone()[0]
    finally:
        connection.close()
    return result.date() if isinstance(result, datetime) else result


def write_log(
    source_name,
    table_name,
    load_type,
    start_time,
    end_time,
    status,
    rows_loaded=0,
    error_message=None,
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
                error_message,
            ),
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

            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]

            print(f"Ошибка временного характера: {error}")

            print(
                f"Повторная попытка {attempt + 2}/{RETRY_ATTEMPTS} через {delay} сек."
            )

            sleep(delay)

    raise last_error


def load_dataset(
    label,
    table,
    loader,
    saver,
    source="CBR",
    date_column="date",
    code=None,
    code_column=None,
):
    """Одинаковые шаги загрузки: период, запрос, сохранение и запись в журнал."""
    started = datetime.now()
    load_type = "incremental"
    try:
        last_date = get_last_date(table, date_column, code_column, code)
        if last_date is None:
            date_from = DEFAULT_START
            load_type = "full"
        else:
            date_from = last_date + timedelta(days=1)
        date_till = date.today()
        rows_loaded = 0
        if date_from <= date_till:
            print(f"\n{label}: {date_from} → {date_till}")
            if code is None:
                frame = retry_call(loader, str(date_from), str(date_till))
            else:
                frame = retry_call(loader, code, str(date_from), str(date_till))
            if not frame.empty:
                saver(frame)
                rows_loaded = len(frame)
        write_log(
            source, table, load_type, started, datetime.now(), "SUCCESS", rows_loaded
        )
        print(f"{label}: загружено {rows_loaded} строк")
        return True
    except Exception as error:
        try:
            write_log(
                source,
                table,
                load_type,
                started,
                datetime.now(),
                "FAILED",
                0,
                str(error),
            )
        except Exception as log_error:
            print(f"{label}: ошибка журнала — {log_error}")
        print(f"{label}: ошибка — {error}")
        return False


def run_moex():
    results = []
    for ticker in TICKERS:
        results.append(
            load_dataset(
                ticker,
                "raw.moex_candles",
                load_data_moex,
                save_to_raw,
                source="MOEX ISS",
                date_column="begin",
                code=ticker,
                code_column="ticker",
            )
        )
    return all(results)


def run_indices():
    results = []
    for code in INDICES:
        results.append(
            load_dataset(
                code,
                "raw.moex_indices",
                load_data_index,
                save_index_to_raw,
                source="MOEX ISS",
                date_column="trading_date",
                code=code,
                code_column="index_code",
            )
        )
    return all(results)


def run_currency():
    results = []
    for currency in CURRENCIES:
        results.append(
            load_dataset(
                currency,
                "raw.cbr_currency",
                load_data_cbr,
                save_currency_to_raw,
                code=currency,
                code_column="currency",
            )
        )
    return all(results)


def run_metals():
    return load_dataset(
        "Металлы", "raw.cbr_metals", load_data_metals, save_metals_to_raw
    )


def run_key_rate():
    return load_dataset(
        "Ключевая ставка", "raw.cbr_key_rate", load_data_key_rate, save_key_rate_to_raw
    )


def run_zcyc():
    return load_dataset(
        "Бескупонная кривая", "raw.cbr_zcyc", load_data_zcyc, save_zcyc_to_raw
    )


def main(argv=None):
    global FULL_RELOAD
    parser = RussianArgumentParser(
        description="Обновление рыночных данных; --full загружает историю с 2020 года"
    )
    parser.add_argument("--full", action="store_true")
    FULL_RELOAD = parser.parse_args(argv).full
    stages = [
        ("Акции MOEX", run_moex),
        ("Индексы MOEX", run_indices),
        ("Валюты", run_currency),
        ("Металлы", run_metals),
        ("Ключевая ставка", run_key_rate),
        ("Бескупонная кривая", run_zcyc),
    ]
    failed = []
    for name, stage in stages:
        print(f"Загрузка: {name}")
        if not stage():
            failed.append(name)
    if failed:
        raise RuntimeError("Загрузка завершилась с ошибками: " + ", ".join(failed))
    print("Загрузка данных завершена")


if __name__ == "__main__":
    main()
