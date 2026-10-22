from datetime import datetime

from db import get_connection


TICKERS = [
    "SBER", "GAZP", "LKOH", "GMKN", "YDEX",
    "ROSN", "NVTK", "TATN", "MGNT", "MTSS",
    "MOEX", "VTBR", "AFLT", "ALRS", "CHMF",
    "NLMK", "PLZL", "PHOR", "SNGS", "SNGSP",
    "TCSG", "OZON", "RUAL", "PIKK", "IRAO"
]

CURRENCIES = [
    "USD", "EUR", "CNY", "GBP", "JPY",
    "CHF", "TRY", "AED", "KZT", "INR"
]

METALS = [
    "GOLD", "SILVER", "PLATINUM", "PALLADIUM"
]

MATURITIES = [
    0.25, 0.5, 0.75, 1.0, 2.0, 3.0,
    5.0, 7.0, 10.0, 15.0, 20.0, 30.0
]

INDICES = ["IMOEX"]


def save_result(
    table,
    check,
    checked,
    bad,
    status,
    details
):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
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
            """,
            (
                table,
                check,
                datetime.now(),
                checked,
                bad,
                status,
                details
            )
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def run_check(table, check, query, details):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            checked, bad = cursor.fetchone()

    finally:
        connection.close()

    status = "PASS" if bad == 0 else "FAIL"

    save_result(
        table,
        check,
        checked,
        bad,
        status,
        details
    )

    print(
        f"{table} | {check} | "
        f"{status} | bad: {bad}"
    )


def check_table(
    table,
    null_condition,
    duplicate_expression,
    positive_condition=None,
    allowed_column=None,
    allowed_values=None,
    freshness_column=None
):
    run_check(
        table,
        "null_values",
        f"""
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE {null_condition}
            )
        FROM {table}
        """,
        "Проверка обязательных полей"
    )

    run_check(
        table,
        "duplicates",
        f"""
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(
                DISTINCT {duplicate_expression}
            )
        FROM {table}
        """,
        "Проверка уникальности записей"
    )

    if positive_condition:
        run_check(
            table,
            "value_range",
            f"""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (
                    WHERE {positive_condition}
                )
            FROM {table}
            """,
            "Проверка допустимого диапазона значений"
        )

    if allowed_column and allowed_values:
        values = ", ".join(
            f"'{value}'"
            for value in allowed_values
        )

        run_check(
            table,
            f"allowed_{allowed_column}",
            f"""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (
                    WHERE {allowed_column}
                    NOT IN ({values})
                )
            FROM {table}
            """,
            f"Проверка допустимых значений {allowed_column}"
        )

    if freshness_column:
        run_check(
            table,
            "freshness",
            f"""
            SELECT
                1,
                CASE
                    WHEN MAX({freshness_column})
                         < CURRENT_DATE - INTERVAL '7 days'
                    THEN 1
                    ELSE 0
                END
            FROM {table}
            """,
            "Проверка актуальности данных"
        )


def check_moex():
    print("\nMOEX АКЦИИ")

    check_table(
        "raw.moex_candles",
        """
        ticker IS NULL
        OR begin IS NULL
        OR open IS NULL
        OR close IS NULL
        OR high IS NULL
        OR low IS NULL
        OR volume IS NULL
        """,
        "(ticker, begin)",
        """
        open <= 0
        OR close <= 0
        OR high <= 0
        OR low <= 0
        OR volume < 0
        """,
        "ticker",
        TICKERS,
        "begin"
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
                   OR open < low
                   OR open > high
            )
        FROM raw.moex_candles
        """,
        "Проверка логики OHLC"
    )

    run_check(
        "raw.moex_candles",
        "freshness_by_ticker",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE last_date < CURRENT_DATE - INTERVAL '7 days'
            )
        FROM (
            SELECT
                ticker,
                MAX(begin::DATE) AS last_date
            FROM raw.moex_candles
            GROUP BY ticker
        ) t
        """,
        "Проверка актуальности по каждому тикеру"
    )


def check_indices():
    print("\nMOEX ИНДЕКСЫ")

    check_table(
        "raw.moex_indices",
        """
        index_code IS NULL
        OR trading_date IS NULL
        OR close IS NULL
        """,
        "(index_code, trading_date)",
        "close <= 0",
        "index_code",
        INDICES,
        "trading_date"
    )


def check_currency():
    print("\nCBR ВАЛЮТЫ")

    check_table(
        "raw.cbr_currency",
        """
        date IS NULL
        OR currency IS NULL
        OR value IS NULL
        """,
        "(date, currency)",
        "value <= 0",
        "currency",
        CURRENCIES,
        "date"
    )

    run_check(
        "raw.cbr_currency",
        "currency_id",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE TRIM(currency_id) = ''
                   OR currency_id IS NULL
            )
        FROM raw.cbr_currency
        """,
        "Проверка идентификатора валюты"
    )

    run_check(
        "raw.cbr_currency",
        "freshness_by_currency",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE last_date < CURRENT_DATE - INTERVAL '7 days'
            )
        FROM (
            SELECT
                currency,
                MAX(date) AS last_date
            FROM raw.cbr_currency
            GROUP BY currency
        ) t
        """,
        "Проверка актуальности по каждой валюте"
    )


def check_metals():
    print("\nCBR МЕТАЛЛЫ")

    check_table(
        "raw.cbr_metals",
        """
        date IS NULL
        OR metal IS NULL
        OR price IS NULL
        """,
        "(date, metal)",
        "price <= 0",
        "metal",
        METALS,
        "date"
    )


def check_key_rate():
    print("\nCBR КЛЮЧЕВАЯ СТАВКА")

    check_table(
        "raw.cbr_key_rate",
        """
        date IS NULL
        OR key_rate IS NULL
        """,
        "date",
        "key_rate <= 0 OR key_rate > 100",
        freshness_column="date"
    )


def check_zcyc():
    print("\nCBR ZCYC")

    check_table(
        "raw.cbr_zcyc",
        """
        date IS NULL
        OR maturity_years IS NULL
        OR yield_percent IS NULL
        """,
        "(date, maturity_years)",
        "yield_percent < 0",
        freshness_column="date"
    )

    run_check(
        "raw.cbr_zcyc",
        "allowed_maturities",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE maturity_years NOT IN (
                    0.25, 0.5, 0.75, 1.0,
                    2.0, 3.0, 5.0, 7.0,
                    10.0, 15.0, 20.0, 30.0
                )
            )
        FROM raw.cbr_zcyc
        """,
        "Проверка допустимых сроков"
    )

    run_check(
        "raw.cbr_zcyc",
        "maturities_per_date",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE maturity_count <> 12
            )
        FROM (
            SELECT
                date,
                COUNT(DISTINCT maturity_years) AS maturity_count
            FROM raw.cbr_zcyc
            GROUP BY date
        ) t
        """,
        "Проверка наличия 12 сроков по каждой дате"
    )


def check_fundamentals():
    print("\nFUNDAMENTALS")

    run_check(
        "core.fundamentals",
        "null_keys",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE instrument_id IS NULL
                   OR ticker IS NULL
                   OR report_year IS NULL
                   OR period_end IS NULL
            )
        FROM core.fundamentals
        """,
        "Проверка обязательных полей"
    )

    run_check(
        "core.fundamentals",
        "duplicates",
        """
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(
                DISTINCT (instrument_id, report_year)
            )
        FROM core.fundamentals
        """,
        "Проверка уникальности instrument_id + report_year"
    )

    run_check(
        "core.fundamentals",
        "invalid_pb",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE pb IS NOT NULL AND pb <= 0
            )
        FROM core.fundamentals
        """,
        "Проверка P/B"
    )

    run_check(
        "core.fundamentals",
        "invalid_market_cap",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE market_cap IS NOT NULL
                  AND market_cap <= 0
            )
        FROM core.fundamentals
        """,
        "Проверка рыночной капитализации"
    )


def check_market_daily():
    print("\nMART MARKET DAILY")

    run_check(
        "mart.market_daily",
        "null_keys",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE ticker IS NULL
                   OR trading_date IS NULL
                   OR close IS NULL
            )
        FROM mart.market_daily
        """,
        "Проверка обязательных полей"
    )

    run_check(
        "mart.market_daily",
        "duplicates",
        """
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(
                DISTINCT (ticker, trading_date)
            )
        FROM mart.market_daily
        """,
        "Проверка уникальности ticker + trading_date"
    )

    run_check(
        "mart.market_daily",
        "invalid_prices",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE close <= 0
                   OR open <= 0
                   OR high <= 0
                   OR low <= 0
            )
        FROM mart.market_daily
        """,
        "Проверка положительных цен"
    )

    run_check(
        "mart.market_daily",
        "price_range",
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE high < low
                   OR close < low
                   OR close > high
                   OR open < low
                   OR open > high
            )
        FROM mart.market_daily
        """,
        "Проверка логики OHLC"
    )


def main():
    print("=" * 70)
    print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("=" * 70)

    check_moex()
    check_indices()
    check_currency()
    check_metals()
    check_key_rate()
    check_zcyc()
    check_fundamentals()
    check_market_daily()

    print("\n" + "=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()