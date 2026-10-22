from datetime import datetime

from db import get_connection


CREATE_FUNDAMENTALS = """
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS core;

CREATE TABLE IF NOT EXISTS staging.finam_fundamentals (
    ticker VARCHAR(20) NOT NULL,
    report_year INTEGER NOT NULL,
    period_end DATE NOT NULL,
    net_income DOUBLE PRECISION,
    assets DOUBLE PRECISION,
    equity DOUBLE PRECISION,
    pb DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    book_to_market DOUBLE PRECISION,
    profitability DOUBLE PRECISION,
    investment DOUBLE PRECISION,
    source_url TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, report_year)
);

CREATE TABLE IF NOT EXISTS core.fundamentals (
    instrument_id INTEGER NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    report_year INTEGER NOT NULL,
    period_end DATE NOT NULL,
    net_income DOUBLE PRECISION,
    assets DOUBLE PRECISION,
    equity DOUBLE PRECISION,
    pb DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    book_to_market DOUBLE PRECISION,
    profitability DOUBLE PRECISION,
    investment DOUBLE PRECISION,
    source_url TEXT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_id, report_year),
    CONSTRAINT fk_fundamentals_instrument
        FOREIGN KEY (instrument_id)
        REFERENCES core.instruments(instrument_id)
);
"""


QUERIES = [
    (
        "staging.moex_candles",
        """
        INSERT INTO staging.moex_candles (
            ticker, trading_date, open, close, high, low, value, volume
        )
        SELECT
            ticker, begin::DATE, open, close, high, low, value, volume
        FROM raw.moex_candles
        WHERE ticker IS NOT NULL
          AND begin IS NOT NULL
          AND open IS NOT NULL
          AND close IS NOT NULL
          AND high IS NOT NULL
          AND low IS NOT NULL
          AND volume IS NOT NULL
          AND open > 0
          AND close > 0
          AND high > 0
          AND low > 0
          AND high >= low
          AND close BETWEEN low AND high
          AND volume >= 0
        ON CONFLICT (ticker, trading_date)
        DO UPDATE SET
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            value = EXCLUDED.value,
            volume = EXCLUDED.volume
        """
    ),
    (
        "staging.moex_indices",
        """
        INSERT INTO staging.moex_indices (
            index_code, trading_date, close
        )
        SELECT
            index_code, trading_date::DATE, close
        FROM raw.moex_indices
        WHERE index_code IS NOT NULL
          AND trading_date IS NOT NULL
          AND close IS NOT NULL
          AND close > 0
        ON CONFLICT (index_code, trading_date)
        DO UPDATE SET close = EXCLUDED.close
        """
    ),
    (
        "staging.cbr_currency",
        """
        INSERT INTO staging.cbr_currency (
            currency, date, currency_id, value
        )
        SELECT
            currency, date::DATE, TRIM(currency_id), value
        FROM raw.cbr_currency
        WHERE currency IS NOT NULL
          AND date IS NOT NULL
          AND value IS NOT NULL
          AND value > 0
        ON CONFLICT (date, currency)
        DO UPDATE SET
            currency_id = EXCLUDED.currency_id,
            value = EXCLUDED.value
        """
    ),
    (
        "staging.cbr_metals",
        """
        INSERT INTO staging.cbr_metals (
            metal, date, metal_id, price
        )
        SELECT
            metal, date::DATE, metal_id, price
        FROM raw.cbr_metals
        WHERE metal IS NOT NULL
          AND date IS NOT NULL
          AND price IS NOT NULL
          AND price > 0
        ON CONFLICT (date, metal)
        DO UPDATE SET
            metal_id = EXCLUDED.metal_id,
            price = EXCLUDED.price
        """
    ),
    (
        "staging.cbr_key_rate",
        """
        INSERT INTO staging.cbr_key_rate (
            date, key_rate
        )
        SELECT
            date::DATE, key_rate
        FROM raw.cbr_key_rate
        WHERE date IS NOT NULL
          AND key_rate >= 0
        ON CONFLICT (date)
        DO UPDATE SET key_rate = EXCLUDED.key_rate
        """
    ),
    (
        "staging.cbr_zcyc",
        """
        INSERT INTO staging.cbr_zcyc (
            date, maturity_years, yield_percent
        )
        SELECT
            date::DATE, maturity_years, yield_percent
        FROM raw.cbr_zcyc
        WHERE date IS NOT NULL
          AND maturity_years IS NOT NULL
          AND yield_percent IS NOT NULL
          AND maturity_years > 0
        ON CONFLICT (date, maturity_years)
        DO UPDATE SET
            yield_percent = EXCLUDED.yield_percent
        """
    ),
    (
        "staging.finam_fundamentals",
        """
        INSERT INTO staging.finam_fundamentals (
            ticker, report_year, period_end,
            net_income, assets, equity,
            pb, market_cap, book_to_market,
            profitability, investment,
            source_url, loaded_at
        )
        SELECT
            ticker, report_year, period_end,
            net_income, assets, equity,
            pb, market_cap, book_to_market,
            profitability, investment,
            source_url, CURRENT_TIMESTAMP
        FROM raw.finam_fundamentals
        WHERE ticker IS NOT NULL
          AND report_year IS NOT NULL
          AND period_end IS NOT NULL
        ON CONFLICT (ticker, report_year)
        DO UPDATE SET
            period_end = EXCLUDED.period_end,
            net_income = EXCLUDED.net_income,
            assets = EXCLUDED.assets,
            equity = EXCLUDED.equity,
            pb = EXCLUDED.pb,
            market_cap = EXCLUDED.market_cap,
            book_to_market = EXCLUDED.book_to_market,
            profitability = EXCLUDED.profitability,
            investment = EXCLUDED.investment,
            source_url = EXCLUDED.source_url,
            loaded_at = CURRENT_TIMESTAMP
        """
    ),
    (
        "core.instruments",
        """
        INSERT INTO core.instruments (
            ticker, instrument_name, market, currency
        )
        VALUES
            ('SBER', 'Сбербанк', 'MOEX', 'RUB'),
            ('GAZP', 'Газпром', 'MOEX', 'RUB'),
            ('LKOH', 'ЛУКОЙЛ', 'MOEX', 'RUB'),
            ('GMKN', 'Норникель', 'MOEX', 'RUB'),
            ('YDEX', 'Яндекс', 'MOEX', 'RUB'),
            ('ROSN', 'Роснефть', 'MOEX', 'RUB'),
            ('NVTK', 'Новатэк', 'MOEX', 'RUB'),
            ('TATN', 'Татнефть', 'MOEX', 'RUB'),
            ('MGNT', 'Магнит', 'MOEX', 'RUB'),
            ('MTSS', 'МТС', 'MOEX', 'RUB'),
            ('MOEX', 'Московская биржа', 'MOEX', 'RUB'),
            ('VTBR', 'ВТБ', 'MOEX', 'RUB'),
            ('AFLT', 'Аэрофлот', 'MOEX', 'RUB'),
            ('ALRS', 'Алроса', 'MOEX', 'RUB'),
            ('CHMF', 'Северсталь', 'MOEX', 'RUB'),
            ('NLMK', 'НЛМК', 'MOEX', 'RUB'),
            ('PLZL', 'Полюс', 'MOEX', 'RUB'),
            ('PHOR', 'ФосАгро', 'MOEX', 'RUB'),
            ('SNGS', 'Сургутнефтегаз', 'MOEX', 'RUB'),
            ('SNGSP', 'Сургутнефтегаз-п', 'MOEX', 'RUB'),
            ('TCSG', 'Т-Технологии', 'MOEX', 'RUB'),
            ('OZON', 'Ozon', 'MOEX', 'RUB'),
            ('RUAL', 'Русал', 'MOEX', 'RUB'),
            ('PIKK', 'ПИК', 'MOEX', 'RUB'),
            ('IRAO', 'Интер РАО', 'MOEX', 'RUB')
        ON CONFLICT (ticker)
        DO UPDATE SET
            instrument_name = EXCLUDED.instrument_name,
            market = EXCLUDED.market,
            currency = EXCLUDED.currency
        """
    ),
    (
        "core.market_prices",
        """
        INSERT INTO core.market_prices (
            instrument_id, trading_date,
            open, close, high, low, value, volume
        )
        SELECT
            i.instrument_id, s.trading_date,
            s.open, s.close, s.high, s.low, s.value, s.volume
        FROM staging.moex_candles s
        JOIN core.instruments i
            ON i.ticker = s.ticker
        ON CONFLICT (instrument_id, trading_date)
        DO UPDATE SET
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            value = EXCLUDED.value,
            volume = EXCLUDED.volume
        """
    ),
    (
        "core.market_indices",
        """
        INSERT INTO core.market_indices (
            index_code, trading_date, close
        )
        SELECT
            index_code, trading_date, close
        FROM staging.moex_indices
        ON CONFLICT (index_code, trading_date)
        DO UPDATE SET close = EXCLUDED.close
        """
    ),
    (
        "core.currency_rates",
        """
        INSERT INTO core.currency_rates (
            currency, date, currency_id, value
        )
        SELECT
            currency, date, currency_id, value
        FROM staging.cbr_currency
        ON CONFLICT (currency, date)
        DO UPDATE SET
            currency_id = EXCLUDED.currency_id,
            value = EXCLUDED.value
        """
    ),
    (
        "core.metal_prices",
        """
        INSERT INTO core.metal_prices (
            metal, date, metal_id, price
        )
        SELECT
            metal, date, metal_id, price
        FROM staging.cbr_metals
        ON CONFLICT (metal, date)
        DO UPDATE SET
            metal_id = EXCLUDED.metal_id,
            price = EXCLUDED.price
        """
    ),
    (
        "core.key_rates",
        """
        INSERT INTO core.key_rates (
            date, key_rate
        )
        SELECT
            date, key_rate
        FROM staging.cbr_key_rate
        ON CONFLICT (date)
        DO UPDATE SET key_rate = EXCLUDED.key_rate
        """
    ),
    (
        "core.risk_free_curve",
        """
        INSERT INTO core.risk_free_curve (
            date, maturity_years, yield_percent
        )
        SELECT
            date, maturity_years, yield_percent
        FROM staging.cbr_zcyc
        ON CONFLICT (date, maturity_years)
        DO UPDATE SET
            yield_percent = EXCLUDED.yield_percent
        """
    ),
    (
        "core.fundamentals",
        """
        INSERT INTO core.fundamentals (
            instrument_id, ticker, report_year, period_end,
            net_income, assets, equity,
            pb, market_cap, book_to_market,
            profitability, investment,
            source_url, loaded_at
        )
        SELECT
            i.instrument_id,
            s.ticker,
            s.report_year,
            s.period_end,
            NULLIF(s.net_income, 'NaN'::DOUBLE PRECISION),
            NULLIF(s.assets, 'NaN'::DOUBLE PRECISION),
            NULLIF(s.equity, 'NaN'::DOUBLE PRECISION),

            CASE
                WHEN s.equity > 0
                 AND s.pb > 0
                 AND s.pb != 'NaN'::DOUBLE PRECISION
                THEN s.pb
            END,

            CASE
                WHEN s.equity > 0
                 AND s.market_cap > 0
                 AND s.market_cap != 'NaN'::DOUBLE PRECISION
                THEN s.market_cap
            END,

            CASE
                WHEN s.equity > 0
                 AND s.book_to_market > 0
                 AND s.book_to_market != 'NaN'::DOUBLE PRECISION
                THEN s.book_to_market
            END,

            CASE
                WHEN s.equity > 0
                 AND s.profitability != 'NaN'::DOUBLE PRECISION
                THEN s.profitability
            END,

            CASE
                WHEN s.investment != 'NaN'::DOUBLE PRECISION
                THEN s.investment
            END,

            s.source_url,
            CURRENT_TIMESTAMP
        FROM staging.finam_fundamentals s
        JOIN core.instruments i
            ON i.ticker = s.ticker
        ON CONFLICT (instrument_id, report_year)
        DO UPDATE SET
            ticker = EXCLUDED.ticker,
            period_end = EXCLUDED.period_end,
            net_income = EXCLUDED.net_income,
            assets = EXCLUDED.assets,
            equity = EXCLUDED.equity,
            pb = EXCLUDED.pb,
            market_cap = EXCLUDED.market_cap,
            book_to_market = EXCLUDED.book_to_market,
            profitability = EXCLUDED.profitability,
            investment = EXCLUDED.investment,
            source_url = EXCLUDED.source_url,
            loaded_at = CURRENT_TIMESTAMP
        """
    ),
    (
        "mart.market_daily",
        """
        INSERT INTO mart.market_daily (
            ticker, trading_date,
            open, close, high, low, volume,
            usd_rate, eur_rate, gold_price, key_rate
        )
        SELECT
            i.ticker,
            mp.trading_date,
            mp.open,
            mp.close,
            mp.high,
            mp.low,
            mp.volume,
            usd.value,
            eur.value,
            gold.price,
            kr.key_rate
        FROM core.market_prices mp
        JOIN core.instruments i
            ON i.instrument_id = mp.instrument_id
        LEFT JOIN core.currency_rates usd
            ON usd.date = mp.trading_date
            AND usd.currency = 'USD'
        LEFT JOIN core.currency_rates eur
            ON eur.date = mp.trading_date
            AND eur.currency = 'EUR'
        LEFT JOIN core.metal_prices gold
            ON gold.date = mp.trading_date
            AND gold.metal = 'GOLD'
        LEFT JOIN core.key_rates kr
            ON kr.date = mp.trading_date
        ON CONFLICT (ticker, trading_date)
        DO UPDATE SET
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            volume = EXCLUDED.volume,
            usd_rate = EXCLUDED.usd_rate,
            eur_rate = EXCLUDED.eur_rate,
            gold_price = EXCLUDED.gold_price,
            key_rate = EXCLUDED.key_rate,
            loaded_at = CURRENT_TIMESTAMP
        """
    )
]


def log_transform(cursor, table, start_time, status, rows, error=None):
    cursor.execute(
        """
        INSERT INTO etl.transform_log (
            table_name, start_time, end_time,
            status, rows_affected, error_message
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            table,
            start_time,
            datetime.now(),
            status,
            rows,
            error
        )
    )


def print_fundamentals_stats(cursor):
    cursor.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT ticker),
            MIN(report_year),
            MAX(report_year),
            COUNT(*) FILTER (WHERE equity <= 0),
            COUNT(*) FILTER (WHERE pb IS NOT NULL),
            COUNT(*) FILTER (WHERE book_to_market IS NOT NULL),
            COUNT(*) FILTER (WHERE profitability IS NOT NULL),
            COUNT(*) FILTER (WHERE investment IS NOT NULL)
        FROM core.fundamentals
        """
    )

    (
        rows,
        companies,
        min_year,
        max_year,
        invalid_equity,
        valid_pb,
        valid_bm,
        valid_profitability,
        valid_investment
    ) = cursor.fetchone()

    print("\nFUNDAMENTALS")
    print(f"Строк: {rows}")
    print(f"Компаний: {companies}")
    print(f"Период: {min_year}–{max_year}")
    print(f"equity <= 0: {invalid_equity}")
    print(f"Валидный P/B: {valid_pb}")
    print(f"Валидный Book-to-Market: {valid_bm}")
    print(f"Валидная Profitability: {valid_profitability}")
    print(f"Investment: {valid_investment}")


def main():
    print("ЗАПУСК TRANSFORM")

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_FUNDAMENTALS)

            for table, query in QUERIES:
                start_time = datetime.now()

                try:
                    cursor.execute(query)
                    rows = cursor.rowcount

                    log_transform(
                        cursor,
                        table,
                        start_time,
                        "SUCCESS",
                        rows
                    )

                    print(f"{table}: {rows}")

                except Exception as error:
                    log_transform(
                        cursor,
                        table,
                        start_time,
                        "FAILED",
                        0,
                        str(error)
                    )
                    raise

            print_fundamentals_stats(cursor)

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    print("TRANSFORM ЗАВЕРШЕН")


if __name__ == "__main__":
    main()