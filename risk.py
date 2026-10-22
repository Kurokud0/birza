import numpy as np
import pandas as pd

from datetime import datetime

from psycopg2.extras import execute_values

from db import get_connection


TRADING_DAYS_PER_YEAR = 252
RISK_FREE_MATURITY = 1.0
VAR_LEVEL = 0.05


def create_tables():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS mart.asset_returns (
                    ticker VARCHAR(20) NOT NULL,
                    trading_date DATE NOT NULL,
                    close_price DOUBLE PRECISION NOT NULL,
                    market_close DOUBLE PRECISION NOT NULL,
                    asset_return DOUBLE PRECISION,
                    market_return DOUBLE PRECISION,
                    risk_free_annual_pct DOUBLE PRECISION,
                    risk_free_daily DOUBLE PRECISION,
                    excess_return DOUBLE PRECISION,
                    market_excess_return DOUBLE PRECISION,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, trading_date)
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_asset_returns_date
                ON mart.asset_returns (trading_date)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_asset_returns_ticker
                ON mart.asset_returns (ticker)
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS mart.asset_risk_metrics (
                    ticker VARCHAR(20) PRIMARY KEY,
                    observations INTEGER NOT NULL,
                    first_date DATE NOT NULL,
                    last_date DATE NOT NULL,
                    cumulative_return DOUBLE PRECISION,
                    annualized_return DOUBLE PRECISION,
                    annualized_volatility DOUBLE PRECISION,
                    sharpe_ratio DOUBLE PRECISION,
                    beta DOUBLE PRECISION,
                    capm_alpha_annual DOUBLE PRECISION,
                    var_95 DOUBLE PRECISION,
                    cvar_95 DOUBLE PRECISION,
                    max_drawdown DOUBLE PRECISION,
                    best_day DOUBLE PRECISION,
                    worst_day DOUBLE PRECISION,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS mart.asset_correlation (
                    ticker_a VARCHAR(20) NOT NULL,
                    ticker_b VARCHAR(20) NOT NULL,
                    correlation DOUBLE PRECISION,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker_a, ticker_b)
                )
                """
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def fetch_dataframe(connection, query, params=None):
    """
    Выполняет SQL-запрос и возвращает результат как DataFrame.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            query,
            params
        )

        rows = cursor.fetchall()
        columns = [
            description[0]
            for description in cursor.description
        ]

    return pd.DataFrame(
        rows,
        columns=columns
    )


def load_market_data():
    connection = get_connection()

    try:
        stock_query = """
            SELECT
                ticker,
                trading_date,
                close
            FROM mart.market_daily
            WHERE ticker IS NOT NULL
              AND trading_date IS NOT NULL
              AND close IS NOT NULL
              AND close > 0
            ORDER BY ticker, trading_date
        """

        index_query = """
            SELECT
                trading_date,
                close AS market_close
            FROM core.market_indices
            WHERE index_code = 'IMOEX'
              AND trading_date IS NOT NULL
              AND close IS NOT NULL
              AND close > 0
            ORDER BY trading_date
        """

        risk_free_query = """
            SELECT
                date,
                yield_percent
            FROM core.risk_free_curve
            WHERE maturity_years = %s
              AND yield_percent IS NOT NULL
            ORDER BY date
        """

        stocks = fetch_dataframe(
            connection,
            stock_query
        )

        market = fetch_dataframe(
            connection,
            index_query
        )

        risk_free = fetch_dataframe(
            connection,
            risk_free_query,
            (RISK_FREE_MATURITY,)
        )

    finally:
        connection.close()

    if stocks.empty:
        raise ValueError(
            "mart.market_daily не содержит данных"
        )

    if market.empty:
        raise ValueError(
            "core.market_indices не содержит IMOEX"
        )

    if risk_free.empty:
        raise ValueError(
            "core.risk_free_curve не содержит 1Y ZCYC"
        )

    stocks["trading_date"] = pd.to_datetime(
        stocks["trading_date"]
    )

    market["trading_date"] = pd.to_datetime(
        market["trading_date"]
    )

    risk_free["date"] = pd.to_datetime(
        risk_free["date"]
    )

    stocks["close"] = pd.to_numeric(
        stocks["close"],
        errors="coerce"
    )

    market["market_close"] = pd.to_numeric(
        market["market_close"],
        errors="coerce"
    )

    risk_free["yield_percent"] = pd.to_numeric(
        risk_free["yield_percent"],
        errors="coerce"
    )

    stocks = stocks.dropna(
        subset=[
            "ticker",
            "trading_date",
            "close"
        ]
    )

    market = market.dropna(
        subset=[
            "trading_date",
            "market_close"
        ]
    )

    risk_free = risk_free.dropna(
        subset=[
            "date",
            "yield_percent"
        ]
    )

    return stocks, market, risk_free


def prepare_returns(
    stocks,
    market,
    risk_free
):
    market = market.sort_values(
        "trading_date"
    ).copy()

    market["market_return"] = (
        market["market_close"]
        .pct_change()
    )

    risk_free = risk_free.copy()

    risk_free["risk_free_annual_pct"] = (
        risk_free["yield_percent"]
    )

    # Перевод годовой ставки в эквивалентную
    # дневную доходность.
    risk_free["risk_free_daily"] = (
        (
            1
            + risk_free["risk_free_annual_pct"] / 100
        )
        ** (1 / TRADING_DAYS_PER_YEAR)
        - 1
    )

    risk_free = risk_free[
        [
            "date",
            "risk_free_annual_pct",
            "risk_free_daily"
        ]
    ].rename(
        columns={
            "date": "trading_date"
        }
    )

    data = stocks.merge(
        market[
            [
                "trading_date",
                "market_close",
                "market_return"
            ]
        ],
        on="trading_date",
        how="inner"
    )

    data = data.merge(
        risk_free,
        on="trading_date",
        how="inner"
    )

    data = data.sort_values(
        [
            "ticker",
            "trading_date"
        ]
    ).reset_index(drop=True)

    data["asset_return"] = (
        data
        .groupby("ticker")["close"]
        .pct_change()
    )

    data["excess_return"] = (
        data["asset_return"]
        - data["risk_free_daily"]
    )

    data["market_excess_return"] = (
        data["market_return"]
        - data["risk_free_daily"]
    )

    data = data.rename(
        columns={
            "close": "close_price"
        }
    )

    return data


def calculate_asset_metrics(
    ticker,
    group
):
    group = group.sort_values(
        "trading_date"
    ).copy()

    returns = group[
        "asset_return"
    ].dropna()

    excess_returns = group[
        "excess_return"
    ].dropna()

    paired = group[
        [
            "excess_return",
            "market_excess_return"
        ]
    ].dropna()

    if len(returns) < 2:
        return None

    cumulative_return = (
        (1 + returns).prod() - 1
    )

    observations = len(returns)

    if cumulative_return > -1:
        annualized_return = (
            (1 + cumulative_return)
            ** (
                TRADING_DAYS_PER_YEAR
                / observations
            )
            - 1
        )
    else:
        annualized_return = np.nan

    daily_volatility = returns.std(
        ddof=1
    )

    annualized_volatility = (
        daily_volatility
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    if (
        len(excess_returns) >= 2
        and excess_returns.std(ddof=1) > 0
    ):
        sharpe_ratio = (
            excess_returns.mean()
            / excess_returns.std(ddof=1)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe_ratio = np.nan

    if (
        len(paired) >= 2
        and paired["market_excess_return"].var(ddof=1) > 0
    ):
        beta = (
            paired["excess_return"].cov(
                paired["market_excess_return"]
            )
            / paired["market_excess_return"].var(
                ddof=1
            )
        )

        capm_alpha_daily = (
            paired["excess_return"]
            - beta * paired["market_excess_return"]
        ).mean()

        capm_alpha_annual = (
            capm_alpha_daily
            * TRADING_DAYS_PER_YEAR
        )

    else:
        beta = np.nan
        capm_alpha_annual = np.nan

    # Исторический VaR.
    var_threshold = np.quantile(
        returns,
        VAR_LEVEL
    )

    var_95 = -var_threshold

    tail_returns = returns[
        returns <= var_threshold
    ]

    if len(tail_returns) > 0:
        cvar_95 = -tail_returns.mean()
    else:
        cvar_95 = np.nan

    # Максимальная просадка.
    wealth = (
        1 + returns
    ).cumprod()

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max - 1
    )

    max_drawdown = drawdown.min()

    best_day = returns.max()
    worst_day = returns.min()

    return {
        "ticker": ticker,
        "observations": int(observations),
        "first_date": group["trading_date"].min().date(),
        "last_date": group["trading_date"].max().date(),
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(
            annualized_return
        ),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "sharpe_ratio": float(
            sharpe_ratio
        ),
        "beta": float(beta),
        "capm_alpha_annual": float(
            capm_alpha_annual
        ),
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "max_drawdown": float(max_drawdown),
        "best_day": float(best_day),
        "worst_day": float(worst_day)
    }


def calculate_metrics(data):
    metrics = []

    for ticker, group in data.groupby(
        "ticker"
    ):
        result = calculate_asset_metrics(
            ticker,
            group
        )

        if result is not None:
            metrics.append(result)

    if not metrics:
        raise ValueError(
            "Не удалось рассчитать риск-метрики"
        )

    return pd.DataFrame(
        metrics
    )


def calculate_correlations(data):
    returns = data[
        [
            "trading_date",
            "ticker",
            "asset_return"
        ]
    ].dropna()

    pivot = returns.pivot(
        index="trading_date",
        columns="ticker",
        values="asset_return"
    )

    correlation_matrix = pivot.corr(
        min_periods=60
    )

    rows = []

    tickers = list(
        correlation_matrix.columns
    )

    for ticker_a in tickers:
        for ticker_b in tickers:

            correlation = correlation_matrix.loc[
                ticker_a,
                ticker_b
            ]

            if pd.notna(correlation):

                rows.append(
                    {
                        "ticker_a": ticker_a,
                        "ticker_b": ticker_b,
                        "correlation": float(
                            correlation
                        )
                    }
                )

    return pd.DataFrame(
        rows
    )


def save_returns(data):
    connection = get_connection()

    query = """
        INSERT INTO mart.asset_returns (
            ticker,
            trading_date,
            close_price,
            market_close,
            asset_return,
            market_return,
            risk_free_annual_pct,
            risk_free_daily,
            excess_return,
            market_excess_return
        )
        VALUES %s
        ON CONFLICT (ticker, trading_date)
        DO UPDATE SET
            close_price =
                EXCLUDED.close_price,
            market_close =
                EXCLUDED.market_close,
            asset_return =
                EXCLUDED.asset_return,
            market_return =
                EXCLUDED.market_return,
            risk_free_annual_pct =
                EXCLUDED.risk_free_annual_pct,
            risk_free_daily =
                EXCLUDED.risk_free_daily,
            excess_return =
                EXCLUDED.excess_return,
            market_excess_return =
                EXCLUDED.market_excess_return,
            calculated_at =
                CURRENT_TIMESTAMP
    """

    columns = [
        "ticker",
        "trading_date",
        "close_price",
        "market_close",
        "asset_return",
        "market_return",
        "risk_free_annual_pct",
        "risk_free_daily",
        "excess_return",
        "market_excess_return"
    ]

    values = []

    for row in data[
        columns
    ].itertuples(
        index=False,
        name=None
    ):

        values.append(
            tuple(
                None if pd.isna(value) else value
                for value in row
            )
        )

    if not values:
        connection.close()

        raise ValueError(
            "Нет данных для сохранения доходностей"
        )

    try:
        with connection.cursor() as cursor:

            execute_values(
                cursor,
                query,
                values,
                page_size=1000
            )

        connection.commit()

        print(
            f"mart.asset_returns: "
            f"{len(values)} строк"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_metrics(metrics):
    connection = get_connection()

    query = """
        INSERT INTO mart.asset_risk_metrics (
            ticker,
            observations,
            first_date,
            last_date,
            cumulative_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            beta,
            capm_alpha_annual,
            var_95,
            cvar_95,
            max_drawdown,
            best_day,
            worst_day
        )
        VALUES %s
        ON CONFLICT (ticker)
        DO UPDATE SET
            observations =
                EXCLUDED.observations,
            first_date =
                EXCLUDED.first_date,
            last_date =
                EXCLUDED.last_date,
            cumulative_return =
                EXCLUDED.cumulative_return,
            annualized_return =
                EXCLUDED.annualized_return,
            annualized_volatility =
                EXCLUDED.annualized_volatility,
            sharpe_ratio =
                EXCLUDED.sharpe_ratio,
            beta =
                EXCLUDED.beta,
            capm_alpha_annual =
                EXCLUDED.capm_alpha_annual,
            var_95 =
                EXCLUDED.var_95,
            cvar_95 =
                EXCLUDED.cvar_95,
            max_drawdown =
                EXCLUDED.max_drawdown,
            best_day =
                EXCLUDED.best_day,
            worst_day =
                EXCLUDED.worst_day,
            calculated_at =
                CURRENT_TIMESTAMP
    """

    columns = [
        "ticker",
        "observations",
        "first_date",
        "last_date",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "beta",
        "capm_alpha_annual",
        "var_95",
        "cvar_95",
        "max_drawdown",
        "best_day",
        "worst_day"
    ]

    values = []

    for row in metrics[
        columns
    ].itertuples(
        index=False,
        name=None
    ):

        values.append(
            tuple(
                None if pd.isna(value) else value
                for value in row
            )
        )

    if not values:
        connection.close()

        raise ValueError(
            "Нет данных для сохранения риск-метрик"
        )

    try:
        with connection.cursor() as cursor:

            execute_values(
                cursor,
                query,
                values,
                page_size=100
            )

        connection.commit()

        print(
            f"mart.asset_risk_metrics: "
            f"{len(values)} строк"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_correlations(correlations):
    connection = get_connection()

    query = """
        INSERT INTO mart.asset_correlation (
            ticker_a,
            ticker_b,
            correlation
        )
        VALUES %s
        ON CONFLICT (ticker_a, ticker_b)
        DO UPDATE SET
            correlation =
                EXCLUDED.correlation,
            calculated_at =
                CURRENT_TIMESTAMP
    """

    columns = [
        "ticker_a",
        "ticker_b",
        "correlation"
    ]

    values = []

    for row in correlations[
        columns
    ].itertuples(
        index=False,
        name=None
    ):

        values.append(
            tuple(
                None if pd.isna(value) else value
                for value in row
            )
        )

    if not values:
        connection.close()

        raise ValueError(
            "Нет данных для сохранения корреляций"
        )

    try:
        with connection.cursor() as cursor:

            execute_values(
                cursor,
                query,
                values,
                page_size=1000
            )

        connection.commit()

        print(
            f"mart.asset_correlation: "
            f"{len(values)} строк"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def print_metrics(metrics):
    output = metrics.copy()

    percent_columns = [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "capm_alpha_annual",
        "var_95",
        "cvar_95",
        "max_drawdown",
        "best_day",
        "worst_day"
    ]

    for column in percent_columns:

        output[column] = (
            output[column] * 100
        ).round(2)

    output["sharpe_ratio"] = (
        output["sharpe_ratio"]
        .round(3)
    )

    output["beta"] = (
        output["beta"]
        .round(3)
    )

    output = output.sort_values(
        "ticker"
    )

    print("\n=== РИСК-МЕТРИКИ ===")

    print(
        output[
            [
                "ticker",
                "observations",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "beta",
                "var_95",
                "cvar_95",
                "max_drawdown"
            ]
        ].to_string(
            index=False
        )
    )


def main():
    start_time = datetime.now()

    print("=" * 80)
    print("ЗАПУСК RISK ANALYTICS")
    print("=" * 80)

    try:
        print("\n[1/5] Создание/проверка аналитических таблиц")

        create_tables()

        print(
            "Таблицы mart проверены"
        )

        print("\n[2/5] Загрузка данных из БД")

        stocks, market, risk_free = (
            load_market_data()
        )

        print(
            f"Акции: {len(stocks)} строк"
        )

        print(
            f"IMOEX: {len(market)} строк"
        )

        print(
            f"ZCYC 1Y: {len(risk_free)} строк"
        )

        print("\n[3/5] Расчёт доходностей")

        data = prepare_returns(
            stocks,
            market,
            risk_free
        )

        print(
            f"Итоговых наблюдений: "
            f"{len(data)}"
        )

        print(
            f"Компаний: "
            f"{data['ticker'].nunique()}"
        )

        print(
            f"Период: "
            f"{data['trading_date'].min().date()} "
            f"→ "
            f"{data['trading_date'].max().date()}"
        )

        print("\n[4/5] Сохранение доходностей и расчёт риска")

        save_returns(
            data
        )

        metrics = calculate_metrics(
            data
        )

        save_metrics(
            metrics
        )

        print_metrics(
            metrics
        )

        print("\n[5/5] Расчёт корреляционной матрицы")

        correlations = calculate_correlations(
            data
        )

        save_correlations(
            correlations
        )

        print(
            f"Корреляций сохранено: "
            f"{len(correlations)}"
        )

        print("\n" + "=" * 80)
        print("RISK ANALYTICS ЗАВЕРШЕН")
        print(
            f"Время начала: "
            f"{start_time}"
        )
        print(
            f"Время окончания: "
            f"{datetime.now()}"
        )
        print("=" * 80)

    except Exception as error:

        print(
            "\nRISK ANALYTICS: ОШИБКА"
        )

        print(
            f"{error}"
        )

        raise


if __name__ == "__main__":
    main()