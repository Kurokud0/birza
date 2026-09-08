from datetime import datetime
from db import get_connection


queries = [
    (
        "staging.moex_candles",
        """
        INSERT INTO staging.moex_candles
            (ticker, trading_date, open, close, high, low, value, volume)
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
        "staging.cbr_currency",
        """
        INSERT INTO staging.cbr_currency
            (currency, date, currency_id, value)
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
        INSERT INTO staging.cbr_metals
            (metal, date, metal_id, price)
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
        INSERT INTO staging.cbr_key_rate
            (date, key_rate)
        SELECT
            date::DATE, key_rate
        FROM raw.cbr_key_rate
        WHERE date IS NOT NULL
          AND key_rate >= 0
        ON CONFLICT (date)
        DO UPDATE SET
            key_rate = EXCLUDED.key_rate
        """
    ),
    (
        "core.instruments",
        """
        INSERT INTO core.instruments
            (ticker, instrument_name, market, currency)
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
            ('MTSS', 'МТС', 'MOEX', 'RUB')
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
        INSERT INTO core.market_prices
            (instrument_id, trading_date, open, close, high, low, value, volume)
        SELECT
            i.instrument_id,
            s.trading_date,
            s.open,
            s.close,
            s.high,
            s.low,
            s.value,
            s.volume
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
        "core.currency_rates",
        """
        INSERT INTO core.currency_rates
            (currency, date, currency_id, value)
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
        INSERT INTO core.metal_prices
            (metal, date, metal_id, price)
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
        INSERT INTO core.key_rates
            (date, key_rate)
        SELECT
            date, key_rate
        FROM staging.cbr_key_rate
        ON CONFLICT (date)
        DO UPDATE SET
            key_rate = EXCLUDED.key_rate
        """
    ),
    (
        "mart.market_daily",
        """
        INSERT INTO mart.market_daily
            (trading_date, open, close, high, low, volume,
             usd_rate, eur_rate, gold_price, key_rate)
        SELECT
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
        WHERE i.ticker = 'SBER'
        ON CONFLICT (trading_date)
        DO UPDATE SET
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            volume = EXCLUDED.volume,
            usd_rate = EXCLUDED.usd_rate,
            eur_rate = EXCLUDED.eur_rate,
            gold_price = EXCLUDED.gold_price,
            key_rate = EXCLUDED.key_rate
        """
    )
]


def log_transform(cursor, table, start_time, status, rows_affected, error_message=None):
    cursor.execute(
        """
        INSERT INTO etl.transform_log
            (table_name, start_time, end_time, status, rows_affected, error_message)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        """,
        (
            table,
            start_time,
            datetime.now(),
            status,
            rows_affected,
            error_message
        )
    )


def main():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            for table, query in queries:
                start_time = datetime.now()

                try:
                    cursor.execute(query)
                    rows_affected = cursor.rowcount

                    log_transform(
                        cursor,
                        table,
                        start_time,
                        "SUCCESS",
                        rows_affected
                    )

                    print(f"{table}: {rows_affected}")

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

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()