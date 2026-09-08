from datetime import datetime, timedelta
from db import get_connection


def save_result(
    table_name,
    check_name,
    checked_rows,
    bad_rows,
    status,
    details
):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO etl.data_quality_log (
            table_name,
            check_name,
            check_time,
            checked_rows,
            bad_rows,
            status,
            details
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            table_name,
            check_name,
            datetime.now(),
            checked_rows,
            bad_rows,
            status,
            details
        )
    )

    connection.commit()

    cursor.close()
    connection.close()


def run_check(
    table_name,
    check_name,
    query,
    details
):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(query)
    result = cursor.fetchone()

    cursor.close()
    connection.close()

    checked_rows = result[0]
    bad_rows = result[1]

    status = "PASS" if bad_rows == 0 else "FAIL"

    save_result(
        table_name,
        check_name,
        checked_rows,
        bad_rows,
        status,
        details
    )

    print(
        f"{table_name} | "
        f"{check_name} | "
        f"{status} | "
        f"bad: {bad_rows}"
    )


def check_moex():
    print("\n=== MOEX ===")

    run_check(
        "raw.moex_candles",
        "null_values",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE ticker IS NULL
                   OR begin IS NULL
                   OR open IS NULL
                   OR close IS NULL
                   OR high IS NULL
                   OR low IS NULL
                   OR volume IS NULL
            )
        FROM raw.moex_candles
        """,
        "Проверка обязательных полей"
    )

    run_check(
        "raw.moex_candles",
        "duplicates",
        """
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(
                DISTINCT (ticker, begin)
            )
        FROM raw.moex_candles
        """,
        "Проверка уникальности ticker + begin"
    )

    run_check(
        "raw.moex_candles",
        "negative_values",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE open <= 0
                   OR close <= 0
                   OR high <= 0
                   OR low <= 0
                   OR volume < 0
            )
        FROM raw.moex_candles
        """,
        "Проверка положительных цен и неотрицательного объёма"
    )

    run_check(
        "raw.moex_candles",
        "price_range",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE high < low
                   OR close < low
                   OR close > high
            )
        FROM raw.moex_candles
        """,
        "Проверка логики high, low и close"
    )

    run_check(
        "raw.moex_candles",
        "freshness",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE begin::date < CURRENT_DATE - INTERVAL '7 days'
            )
        FROM raw.moex_candles
        WHERE begin = (
            SELECT MAX(begin)
            FROM raw.moex_candles
        )
        """,
        "Проверка наличия актуальных данных"
    )


def check_currency():
    print("\n=== CBR CURRENCY ===")

    run_check(
        "raw.cbr_currency",
        "null_values",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE date IS NULL
                   OR currency IS NULL
                   OR value IS NULL
            )
        FROM raw.cbr_currency
        """,
        "Проверка обязательных полей"
    )

    run_check(
        "raw.cbr_currency",
        "duplicates",
        """
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(
                DISTINCT (date, currency)
            )
        FROM raw.cbr_currency
        """,
        "Проверка уникальности date + currency"
    )

    run_check(
        "raw.cbr_currency",
        "negative_values",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE value <= 0
            )
        FROM raw.cbr_currency
        """,
        "Проверка положительных курсов"
    )

    run_check(
        "raw.cbr_currency",
        "allowed_currencies",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE currency NOT IN (
                    'USD',
                    'EUR',
                    'CNY',
                    'GBP',
                    'JPY',
                    'CHF',
                    'TRY',
                    'AED',
                    'KZT',
                    'INR'
                )
            )
        FROM raw.cbr_currency
        """,
        "Проверка допустимых кодов валют"
    )


def check_metals():
    print("\n=== CBR METALS ===")

    run_check(
        "raw.cbr_metals",
        "null_values",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE date IS NULL
                   OR metal IS NULL
                   OR price IS NULL
            )
        FROM raw.cbr_metals
        """,
        "Проверка обязательных полей"
    )

    run_check(
        "raw.cbr_metals",
        "duplicates",
        """
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(
                DISTINCT (date, metal)
            )
        FROM raw.cbr_metals
        """,
        "Проверка уникальности date + metal"
    )

    run_check(
        "raw.cbr_metals",
        "negative_values",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE price <= 0
            )
        FROM raw.cbr_metals
        """,
        "Проверка положительных цен"
    )

    run_check(
        "raw.cbr_metals",
        "allowed_metals",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE metal NOT IN (
                    'GOLD',
                    'SILVER',
                    'PLATINUM',
                    'PALLADIUM'
                )
            )
        FROM raw.cbr_metals
        """,
        "Проверка допустимых металлов"
    )


def check_key_rate():
    print("\n=== CBR KEY RATE ===")

    run_check(
        "raw.cbr_key_rate",
        "null_values",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE date IS NULL
                   OR key_rate IS NULL
            )
        FROM raw.cbr_key_rate
        """,
        "Проверка обязательных полей"
    )

    run_check(
        "raw.cbr_key_rate",
        "duplicates",
        """
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(
                DISTINCT date
            )
        FROM raw.cbr_key_rate
        """,
        "Проверка уникальности даты"
    )

    run_check(
        "raw.cbr_key_rate",
        "allowed_range",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE key_rate <= 0
                   OR key_rate > 100
            )
        FROM raw.cbr_key_rate
        """,
        "Проверка допустимого диапазона ключевой ставки"
    )


def main():
    print("ЗАПУСК ПРОВЕРКИ КАЧЕСТВА ДАННЫХ")
    print(f"Время запуска: {datetime.now()}")

    check_moex()
    check_currency()
    check_metals()
    check_key_rate()

    print("\nПРОВЕРКА ЗАВЕРШЕНА")


if __name__ == "__main__":
    main()
