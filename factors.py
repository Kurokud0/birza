from datetime import datetime
import math

import pandas as pd

from db import get_connection


# ============================================================
# НАСТРОЙКИ
# ============================================================

TRADING_DAYS = 252

# Отчётный год t используется с 1 июля t+1.
EFFECTIVE_MONTH = 7
EFFECTIVE_DAY = 1

# Используем данные начиная с 2021 года.
START_YEAR = 2021


# ============================================================
# СОЗДАНИЕ ТАБЛИЦ
# ============================================================

CREATE_TABLES_SQL = """
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS mart.ff5_assignments (
    effective_date DATE NOT NULL,
    end_date DATE NOT NULL,

    ticker VARCHAR(20) NOT NULL,
    report_year INTEGER NOT NULL,

    market_cap DOUBLE PRECISION,
    book_to_market DOUBLE PRECISION,
    profitability DOUBLE PRECISION,
    investment DOUBLE PRECISION,

    size_group VARCHAR(1),

    value_group VARCHAR(1),
    profitability_group VARCHAR(1),
    investment_group VARCHAR(1),

    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (
        effective_date,
        ticker
    )
);


CREATE TABLE IF NOT EXISTS mart.ff5_factors (
    trading_date DATE PRIMARY KEY,

    report_year INTEGER,

    rf DOUBLE PRECISION,
    market_return DOUBLE PRECISION,
    mkt_rf DOUBLE PRECISION,

    smb DOUBLE PRECISION,
    hml DOUBLE PRECISION,
    rmw DOUBLE PRECISION,
    cma DOUBLE PRECISION,

    smb_bm DOUBLE PRECISION,
    smb_profitability DOUBLE PRECISION,
    smb_investment DOUBLE PRECISION,

    bm_sl DOUBLE PRECISION,
    bm_sn DOUBLE PRECISION,
    bm_sh DOUBLE PRECISION,

    bm_bl DOUBLE PRECISION,
    bm_bn DOUBLE PRECISION,
    bm_bh DOUBLE PRECISION,

    op_sw DOUBLE PRECISION,
    op_sn DOUBLE PRECISION,
    op_sr DOUBLE PRECISION,

    op_bw DOUBLE PRECISION,
    op_bn DOUBLE PRECISION,
    op_br DOUBLE PRECISION,

    inv_sa DOUBLE PRECISION,
    inv_sn DOUBLE PRECISION,
    inv_sc DOUBLE PRECISION,

    inv_ba DOUBLE PRECISION,
    inv_bn DOUBLE PRECISION,
    inv_bc DOUBLE PRECISION,

    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ============================================================
# SQL VALUE
# ============================================================

def sql_value(value):
    """
    Преобразует pandas NaN/NaT в SQL NULL.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


# ============================================================
# SQL QUERY -> DATAFRAME
# ============================================================

def read_sql_dataframe(
    connection,
    query,
    params=None,
):
    """
    Чтение PostgreSQL через cursor.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            query,
            params or (),
        )

        rows = cursor.fetchall()

        columns = [
            description[0]
            for description in cursor.description
        ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# ============================================================
# ПРИВЕДЕНИЕ ЧИСЛОВЫХ СТОЛБЦОВ
# ============================================================

def convert_numeric_columns(
    df,
    columns,
):
    """
    PostgreSQL может вернуть NUMERIC как Decimal.
    Приводим всё к float перед математическими операциями.
    """

    if df.empty:
        return df

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype(float)

    return df


# ============================================================
# ЗАГРУЗКА FUNDAMENTALS
# ============================================================

def load_fundamentals(connection):

    query = """
        SELECT
            ticker,
            report_year,
            period_end,

            market_cap,
            book_to_market,
            profitability,
            investment

        FROM core.fundamentals

        WHERE report_year >= %s

        ORDER BY
            report_year,
            ticker
    """

    df = read_sql_dataframe(
        connection,
        query,
        (START_YEAR,),
    )

    df = convert_numeric_columns(
        df,
        [
            "market_cap",
            "book_to_market",
            "profitability",
            "investment",
        ],
    )

    if not df.empty:

        df["report_year"] = pd.to_numeric(
            df["report_year"],
            errors="coerce",
        ).astype("Int64")

    return df


# ============================================================
# ЗАГРУЗКА ЦЕН
# ============================================================

def load_prices(connection):

    query = """
        SELECT
            i.ticker,
            mp.trading_date,
            mp.close

        FROM core.market_prices mp

        JOIN core.instruments i
            ON i.instrument_id = mp.instrument_id

        WHERE mp.close IS NOT NULL
          AND mp.close > 0

        ORDER BY
            mp.trading_date,
            i.ticker
    """

    df = read_sql_dataframe(
        connection,
        query,
    )

    df = convert_numeric_columns(
        df,
        [
            "close",
        ],
    )

    if not df.empty:

        df["trading_date"] = pd.to_datetime(
            df["trading_date"]
        )

        df = df.sort_values(
            [
                "ticker",
                "trading_date",
            ]
        )

        # Дневная доходность акции.
        df["return"] = (
            df.groupby("ticker")["close"]
            .pct_change()
        )

        # На всякий случай ещё раз float.
        df["return"] = pd.to_numeric(
            df["return"],
            errors="coerce",
        ).astype(float)

    return df


# ============================================================
# ЗАГРУЗКА IMOEX
# ============================================================

def load_market_index(connection):

    query = """
        SELECT
            trading_date,
            close

        FROM core.market_indices

        WHERE index_code = 'IMOEX'
          AND close IS NOT NULL
          AND close > 0

        ORDER BY
            trading_date
    """

    df = read_sql_dataframe(
        connection,
        query,
    )

    df = convert_numeric_columns(
        df,
        [
            "close",
        ],
    )

    if not df.empty:

        df["trading_date"] = pd.to_datetime(
            df["trading_date"]
        )

        df = df.sort_values(
            "trading_date"
        )

        df["market_return"] = (
            df["close"]
            .pct_change()
        )

        df["market_return"] = pd.to_numeric(
            df["market_return"],
            errors="coerce",
        ).astype(float)

    return df


# ============================================================
# ЗАГРУЗКА RISK FREE
# ============================================================

def load_risk_free(connection):

    query = """
        SELECT
            date,
            maturity_years,
            yield_percent

        FROM core.risk_free_curve

        WHERE maturity_years = 1.0
          AND yield_percent IS NOT NULL

        ORDER BY
            date
    """

    df = read_sql_dataframe(
        connection,
        query,
    )

    # ========================================================
    # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
    # ========================================================
    #
    # yield_percent мог прийти из PostgreSQL
    # как decimal.Decimal.
    #
    # Приводим к float до выполнения
    # арифметических операций.
    # ========================================================

    df = convert_numeric_columns(
        df,
        [
            "maturity_years",
            "yield_percent",
        ],
    )

    if not df.empty:

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df = df.sort_values(
            "date"
        )

        # Годовая доходность ZCYC
        # -> эффективная дневная ставка.

        df["rf"] = (
            (
                1.0
                + df["yield_percent"] / 100.0
            )
            ** (1.0 / TRADING_DAYS)
            - 1.0
        )

        df["rf"] = pd.to_numeric(
            df["rf"],
            errors="coerce",
        ).astype(float)

    return df


# ============================================================
# ПЕРИОДЫ ПЕРЕБАЛАНСИРОВКИ
# ============================================================

def build_rebalance_periods(
    fundamentals,
    max_trading_date,
):

    periods = []

    if fundamentals.empty:
        return periods

    report_years = sorted(
        fundamentals["report_year"]
        .dropna()
        .astype(int)
        .unique()
    )

    for report_year in report_years:

        if report_year < START_YEAR:
            continue

        effective_date = pd.Timestamp(
            year=report_year + 1,
            month=EFFECTIVE_MONTH,
            day=EFFECTIVE_DAY,
        )

        next_effective_date = pd.Timestamp(
            year=report_year + 2,
            month=EFFECTIVE_MONTH,
            day=EFFECTIVE_DAY,
        )

        end_date = (
            next_effective_date
            - pd.Timedelta(days=1)
        )

        if effective_date > max_trading_date:
            continue

        actual_end = min(
            end_date,
            max_trading_date,
        )

        periods.append(
            {
                "report_year": report_year,
                "effective_date": effective_date,
                "end_date": actual_end,
            }
        )

    return periods


# ============================================================
# SIZE GROUP
# ============================================================

def add_size_group(df):

    df = df.copy()

    df["size_group"] = None

    valid = (
        df["market_cap"].notna()
        & (df["market_cap"] > 0)
    )

    if valid.sum() == 0:
        return df

    median_market_cap = (
        df.loc[
            valid,
            "market_cap",
        ]
        .median()
    )

    df.loc[
        valid
        & (
            df["market_cap"]
            <= median_market_cap
        ),
        "size_group",
    ] = "S"

    df.loc[
        valid
        & (
            df["market_cap"]
            > median_market_cap
        ),
        "size_group",
    ] = "B"

    return df


# ============================================================
# FACTOR GROUP
# ============================================================

def add_factor_group(
    df,
    factor_column,
    group_column,
    low_label,
    neutral_label,
    high_label,
):

    df = df.copy()

    df[group_column] = None

    valid = (
        df["market_cap"].notna()
        & (df["market_cap"] > 0)
        & df["size_group"].notna()
        & df[factor_column].notna()
    )

    if valid.sum() == 0:
        return df

    # --------------------------------------------------------
    # Breakpoints преимущественно считаем
    # по BIG stocks.
    # --------------------------------------------------------

    big_valid = (
        valid
        & (df["size_group"] == "B")
    )

    big_values = (
        df.loc[
            big_valid,
            factor_column,
        ]
        .dropna()
    )

    if (
        len(big_values) >= 6
        and big_values.nunique() >= 3
    ):

        breakpoint_values = big_values

    else:

        breakpoint_values = (
            df.loc[
                valid,
                factor_column,
            ]
            .dropna()
        )

    if breakpoint_values.empty:
        return df

    low_breakpoint = (
        breakpoint_values
        .quantile(0.30)
    )

    high_breakpoint = (
        breakpoint_values
        .quantile(0.70)
    )

    # Если границы совпали —
    # используем трети.

    if low_breakpoint >= high_breakpoint:

        low_breakpoint = (
            breakpoint_values
            .quantile(1.0 / 3.0)
        )

        high_breakpoint = (
            breakpoint_values
            .quantile(2.0 / 3.0)
        )

    # LOW

    df.loc[
        valid
        & (
            df[factor_column]
            <= low_breakpoint
        ),
        group_column,
    ] = low_label

    # HIGH

    df.loc[
        valid
        & (
            df[factor_column]
            >= high_breakpoint
        ),
        group_column,
    ] = high_label

    # NEUTRAL

    df.loc[
        valid
        & (
            df[factor_column]
            > low_breakpoint
        )
        & (
            df[factor_column]
            < high_breakpoint
        ),
        group_column,
    ] = neutral_label

    return df


# ============================================================
# ASSIGNMENTS ДЛЯ ПЕРИОДА
# ============================================================

def create_assignments_for_period(
    fundamentals,
    report_year,
):

    df = fundamentals[
        fundamentals["report_year"]
        == report_year
    ].copy()

    if df.empty:
        return df

    # SIZE

    df = add_size_group(
        df
    )

    # VALUE

    df = add_factor_group(
        df=df,
        factor_column="book_to_market",
        group_column="value_group",
        low_label="L",
        neutral_label="N",
        high_label="H",
    )

    # PROFITABILITY

    df = add_factor_group(
        df=df,
        factor_column="profitability",
        group_column="profitability_group",
        low_label="W",
        neutral_label="N",
        high_label="R",
    )

    # INVESTMENT

    df = add_factor_group(
        df=df,
        factor_column="investment",
        group_column="investment_group",
        low_label="A",
        neutral_label="N",
        high_label="C",
    )

    return df


# ============================================================
# VALUE-WEIGHTED RETURN
# ============================================================

def weighted_return(group):

    group = group.dropna(
        subset=[
            "return",
            "market_cap",
        ]
    )

    group = group[
        group["market_cap"] > 0
    ]

    if group.empty:
        return None

    denominator = (
        group["market_cap"]
        .sum()
    )

    if denominator <= 0:
        return None

    numerator = (
        group["return"]
        * group["market_cap"]
    ).sum()

    return (
        numerator
        / denominator
    )


# ============================================================
# PORTFOLIO RETURNS
# ============================================================

def calculate_portfolio_returns(
    daily_returns,
    factor_group_column,
):

    required_columns = [
        "trading_date",
        "return",
        "market_cap",
        "size_group",
        factor_group_column,
    ]

    data = daily_returns[
        required_columns
    ].copy()

    data = data[
        data["return"].notna()
        & data["market_cap"].notna()
        & data["size_group"].notna()
        & data[factor_group_column].notna()
        & (data["market_cap"] > 0)
    ]

    if data.empty:
        return pd.DataFrame()

    grouped = (
        data
        .groupby(
            [
                "trading_date",
                "size_group",
                factor_group_column,
            ],
            dropna=False,
        )
        .apply(
            weighted_return,
            include_groups=False,
        )
        .reset_index(
            name="portfolio_return"
        )
    )

    pivot = (
        grouped
        .pivot(
            index="trading_date",
            columns=[
                "size_group",
                factor_group_column,
            ],
            values="portfolio_return",
        )
    )

    return pivot


# ============================================================
# FLATTEN COLUMNS
# ============================================================

def flatten_portfolio_columns(
    pivot,
    prefix,
):

    if pivot.empty:
        return pivot

    result = pivot.copy()

    new_columns = []

    for column in result.columns:

        if isinstance(column, tuple):

            size = column[0]
            group = column[1]

            new_columns.append(
                f"{prefix}_{size}{group}".lower()
            )

        else:

            new_columns.append(
                f"{prefix}_{column}".lower()
            )

    result.columns = new_columns

    return result.reset_index()


# ============================================================
# AVAILABLE MEAN
# ============================================================

def available_mean(
    dataframe,
    columns,
    minimum=1,
):
    """
    Среднее по доступным портфелям.

    Если одна ячейка отсутствует,
    используем остальные доступные.

    minimum — минимальное количество
    доступных портфелей.
    """

    existing = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    if not existing:

        return pd.Series(
            index=dataframe.index,
            dtype="float64",
        )

    values = dataframe[
        existing
    ]

    count = (
        values
        .notna()
        .sum(axis=1)
    )

    result = values.mean(
        axis=1,
        skipna=True,
    )

    result[count < minimum] = float(
        "nan"
    )

    return result


# ============================================================
# CALCULATE FACTORS
# ============================================================

def calculate_factors_for_period(
    daily_returns,
    period_assignments,
):

    if daily_returns.empty:
        return pd.DataFrame()

    assignment_columns = [
        "ticker",
        "market_cap",
        "size_group",
        "value_group",
        "profitability_group",
        "investment_group",
    ]

    merged = daily_returns.merge(
        period_assignments[
            assignment_columns
        ],
        on="ticker",
        how="left",
    )

    merged = merged[
        merged["return"].notna()
    ]

    if merged.empty:
        return pd.DataFrame()

    # ========================================================
    # B/M
    # ========================================================

    bm_pivot = calculate_portfolio_returns(
        merged,
        "value_group",
    )

    bm_pivot = flatten_portfolio_columns(
        bm_pivot,
        "bm",
    )

    # ========================================================
    # PROFITABILITY
    # ========================================================

    op_pivot = calculate_portfolio_returns(
        merged,
        "profitability_group",
    )

    op_pivot = flatten_portfolio_columns(
        op_pivot,
        "op",
    )

    # ========================================================
    # INVESTMENT
    # ========================================================

    inv_pivot = calculate_portfolio_returns(
        merged,
        "investment_group",
    )

    inv_pivot = flatten_portfolio_columns(
        inv_pivot,
        "inv",
    )

    # ========================================================
    # ОБЪЕДИНЕНИЕ
    # ========================================================

    result = None

    for dataframe in [
        bm_pivot,
        op_pivot,
        inv_pivot,
    ]:

        if dataframe.empty:
            continue

        if result is None:

            result = dataframe.copy()

        else:

            result = result.merge(
                dataframe,
                on="trading_date",
                how="outer",
            )

    if result is None or result.empty:
        return pd.DataFrame()

    result = (
        result
        .sort_values("trading_date")
        .reset_index(drop=True)
    )

    # ========================================================
    # HML
    # ========================================================

    result["smb_bm"] = (
        available_mean(
            result,
            [
                "bm_sl",
                "bm_sn",
                "bm_sh",
            ],
            minimum=1,
        )
        -
        available_mean(
            result,
            [
                "bm_bl",
                "bm_bn",
                "bm_bh",
            ],
            minimum=1,
        )
    )

    result["hml"] = (
        available_mean(
            result,
            [
                "bm_sh",
                "bm_bh",
            ],
            minimum=1,
        )
        -
        available_mean(
            result,
            [
                "bm_sl",
                "bm_bl",
            ],
            minimum=1,
        )
    )

    # ========================================================
    # RMW
    # ========================================================

    result["smb_profitability"] = (
        available_mean(
            result,
            [
                "op_sw",
                "op_sn",
                "op_sr",
            ],
            minimum=1,
        )
        -
        available_mean(
            result,
            [
                "op_bw",
                "op_bn",
                "op_br",
            ],
            minimum=1,
        )
    )

    result["rmw"] = (
        available_mean(
            result,
            [
                "op_sr",
                "op_br",
            ],
            minimum=1,
        )
        -
        available_mean(
            result,
            [
                "op_sw",
                "op_bw",
            ],
            minimum=1,
        )
    )

    # ========================================================
    # CMA
    # ========================================================

    result["smb_investment"] = (
        available_mean(
            result,
            [
                "inv_sa",
                "inv_sn",
                "inv_sc",
            ],
            minimum=1,
        )
        -
        available_mean(
            result,
            [
                "inv_ba",
                "inv_bn",
                "inv_bc",
            ],
            minimum=1,
        )
    )

    result["cma"] = (
        available_mean(
            result,
            [
                "inv_sc",
                "inv_bc",
            ],
            minimum=1,
        )
        -
        available_mean(
            result,
            [
                "inv_sa",
                "inv_ba",
            ],
            minimum=1,
        )
    )

    # ========================================================
    # SMB
    # ========================================================

    smb_components = result[
        [
            "smb_bm",
            "smb_profitability",
            "smb_investment",
        ]
    ]

    smb_count = (
        smb_components
        .notna()
        .sum(axis=1)
    )

    result["smb"] = (
        smb_components
        .mean(
            axis=1,
            skipna=True,
        )
    )

    # Минимум две компоненты.
    result.loc[
        smb_count < 2,
        "smb",
    ] = float(
        "nan"
    )

    return result


# ============================================================
# ВСЕ ASSIGNMENTS
# ============================================================

def build_all_assignments(
    fundamentals,
    periods,
):

    assignments = []

    for period in periods:

        report_year = period[
            "report_year"
        ]

        effective_date = period[
            "effective_date"
        ]

        end_date = period[
            "end_date"
        ]

        current = create_assignments_for_period(
            fundamentals,
            report_year,
        )

        if current.empty:
            continue

        current[
            "effective_date"
        ] = effective_date

        current[
            "end_date"
        ] = end_date

        assignments.append(
            current
        )

    if not assignments:
        return pd.DataFrame()

    result = pd.concat(
        assignments,
        ignore_index=True,
    )

    return result


# ============================================================
# СОХРАНЕНИЕ ASSIGNMENTS
# ============================================================

def save_assignments(
    connection,
    assignments,
):

    if assignments.empty:
        return 0

    sql = """
        INSERT INTO mart.ff5_assignments (
            effective_date,
            end_date,

            ticker,
            report_year,

            market_cap,
            book_to_market,
            profitability,
            investment,

            size_group,

            value_group,
            profitability_group,
            investment_group,

            loaded_at
        )
        VALUES (
            %s,
            %s,

            %s,
            %s,

            %s,
            %s,
            %s,
            %s,

            %s,

            %s,
            %s,
            %s,

            CURRENT_TIMESTAMP
        )

        ON CONFLICT (
            effective_date,
            ticker
        )
        DO UPDATE SET

            end_date = EXCLUDED.end_date,
            report_year = EXCLUDED.report_year,

            market_cap = EXCLUDED.market_cap,
            book_to_market = EXCLUDED.book_to_market,
            profitability = EXCLUDED.profitability,
            investment = EXCLUDED.investment,

            size_group = EXCLUDED.size_group,

            value_group = EXCLUDED.value_group,
            profitability_group =
                EXCLUDED.profitability_group,
            investment_group =
                EXCLUDED.investment_group,

            loaded_at = CURRENT_TIMESTAMP
    """

    rows = []

    for row in assignments.itertuples(
        index=False
    ):

        rows.append(
            (
                row.effective_date.date(),
                row.end_date.date(),

                row.ticker,
                int(row.report_year),

                sql_value(row.market_cap),
                sql_value(row.book_to_market),
                sql_value(row.profitability),
                sql_value(row.investment),

                sql_value(row.size_group),

                sql_value(row.value_group),
                sql_value(row.profitability_group),
                sql_value(row.investment_group),
            )
        )

    with connection.cursor() as cursor:

        cursor.executemany(
            sql,
            rows,
        )

    return len(rows)


# ============================================================
# ВСЕ FACTORS
# ============================================================

def build_all_factors(
    prices,
    market_index,
    risk_free,
    periods,
    assignments,
):

    if prices.empty:
        return pd.DataFrame()

    all_factors = []

    for period in periods:

        report_year = period[
            "report_year"
        ]

        effective_date = period[
            "effective_date"
        ]

        end_date = period[
            "end_date"
        ]

        current_assignments = assignments[
            assignments["report_year"]
            == report_year
        ].copy()

        if current_assignments.empty:
            continue

        current_prices = prices[
            (prices["trading_date"] >= effective_date)
            & (prices["trading_date"] <= end_date)
        ].copy()

        if current_prices.empty:
            continue

        current_factors = calculate_factors_for_period(
            current_prices,
            current_assignments,
        )

        if current_factors.empty:
            continue

        current_factors[
            "report_year"
        ] = report_year

        all_factors.append(
            current_factors
        )

    if not all_factors:
        return pd.DataFrame()

    result = pd.concat(
        all_factors,
        ignore_index=True,
    )

    result = (
        result
        .sort_values("trading_date")
        .reset_index(drop=True)
    )

    # ========================================================
    # IMOEX
    # ========================================================

    market = market_index[
        [
            "trading_date",
            "market_return",
        ]
    ].copy()

    market = market.sort_values(
        "trading_date"
    )

    result = result.sort_values(
        "trading_date"
    )

    result = pd.merge_asof(
        result,
        market,
        on="trading_date",
        direction="backward",
    )

    # ========================================================
    # RISK FREE
    # ========================================================

    rf = risk_free[
        [
            "date",
            "rf",
        ]
    ].copy()

    rf = rf.rename(
        columns={
            "date": "trading_date",
        }
    )

    rf = rf.sort_values(
        "trading_date"
    )

    result = result.sort_values(
        "trading_date"
    )

    result = pd.merge_asof(
        result,
        rf,
        on="trading_date",
        direction="backward",
    )

    # ========================================================
    # MKT-RF
    # ========================================================

    result["mkt_rf"] = (
        result["market_return"]
        - result["rf"]
    )

    return result


# ============================================================
# CLEAN FACTORS
# ============================================================

def clean_factors(df):

    if df.empty:
        return df

    numeric_columns = [
        "rf",
        "market_return",
        "mkt_rf",

        "smb",
        "hml",
        "rmw",
        "cma",

        "smb_bm",
        "smb_profitability",
        "smb_investment",

        "bm_sl",
        "bm_sn",
        "bm_sh",

        "bm_bl",
        "bm_bn",
        "bm_bh",

        "op_sw",
        "op_sn",
        "op_sr",

        "op_bw",
        "op_bn",
        "op_br",

        "inv_sa",
        "inv_sn",
        "inv_sc",

        "inv_ba",
        "inv_bn",
        "inv_bc",
    ]

    df = convert_numeric_columns(
        df,
        numeric_columns,
    )

    return df


# ============================================================
# СОХРАНЕНИЕ FACTORS
# ============================================================

def save_factors(
    connection,
    factors,
):

    if factors.empty:
        return 0

    sql = """
        INSERT INTO mart.ff5_factors (
            trading_date,
            report_year,

            rf,
            market_return,
            mkt_rf,

            smb,
            hml,
            rmw,
            cma,

            smb_bm,
            smb_profitability,
            smb_investment,

            bm_sl,
            bm_sn,
            bm_sh,

            bm_bl,
            bm_bn,
            bm_bh,

            op_sw,
            op_sn,
            op_sr,

            op_bw,
            op_bn,
            op_br,

            inv_sa,
            inv_sn,
            inv_sc,

            inv_ba,
            inv_bn,
            inv_bc,

            loaded_at
        )
        VALUES (
            %s,
            %s,

            %s,
            %s,
            %s,

            %s,
            %s,
            %s,
            %s,

            %s,
            %s,
            %s,

            %s,
            %s,
            %s,

            %s,
            %s,
            %s,

            %s,
            %s,
            %s,

            %s,
            %s,
            %s,

            %s,
            %s,
            %s,

            %s,
            %s,
            %s,

            CURRENT_TIMESTAMP
        )

        ON CONFLICT (
            trading_date
        )
        DO UPDATE SET

            report_year = EXCLUDED.report_year,

            rf = EXCLUDED.rf,
            market_return = EXCLUDED.market_return,
            mkt_rf = EXCLUDED.mkt_rf,

            smb = EXCLUDED.smb,
            hml = EXCLUDED.hml,
            rmw = EXCLUDED.rmw,
            cma = EXCLUDED.cma,

            smb_bm = EXCLUDED.smb_bm,
            smb_profitability =
                EXCLUDED.smb_profitability,
            smb_investment =
                EXCLUDED.smb_investment,

            bm_sl = EXCLUDED.bm_sl,
            bm_sn = EXCLUDED.bm_sn,
            bm_sh = EXCLUDED.bm_sh,

            bm_bl = EXCLUDED.bm_bl,
            bm_bn = EXCLUDED.bm_bn,
            bm_bh = EXCLUDED.bm_bh,

            op_sw = EXCLUDED.op_sw,
            op_sn = EXCLUDED.op_sn,
            op_sr = EXCLUDED.op_sr,

            op_bw = EXCLUDED.op_bw,
            op_bn = EXCLUDED.op_bn,
            op_br = EXCLUDED.op_br,

            inv_sa = EXCLUDED.inv_sa,
            inv_sn = EXCLUDED.inv_sn,
            inv_sc = EXCLUDED.inv_sc,

            inv_ba = EXCLUDED.inv_ba,
            inv_bn = EXCLUDED.inv_bn,
            inv_bc = EXCLUDED.inv_bc,

            loaded_at = CURRENT_TIMESTAMP
    """

    columns = [
        "rf",
        "market_return",
        "mkt_rf",

        "smb",
        "hml",
        "rmw",
        "cma",

        "smb_bm",
        "smb_profitability",
        "smb_investment",

        "bm_sl",
        "bm_sn",
        "bm_sh",

        "bm_bl",
        "bm_bn",
        "bm_bh",

        "op_sw",
        "op_sn",
        "op_sr",

        "op_bw",
        "op_bn",
        "op_br",

        "inv_sa",
        "inv_sn",
        "inv_sc",

        "inv_ba",
        "inv_bn",
        "inv_bc",
    ]

    rows = []

    for row in factors.itertuples(
        index=False
    ):

        values = []

        for column in columns:

            values.append(
                sql_value(
                    getattr(
                        row,
                        column,
                    )
                )
            )

        rows.append(
            (
                row.trading_date.date(),
                int(row.report_year),
                *values,
            )
        )

    with connection.cursor() as cursor:

        cursor.executemany(
            sql,
            rows,
        )

    return len(rows)


# ============================================================
# СТАТИСТИКА FACTORS
# ============================================================

def print_factor_statistics(
    factors,
):

    if factors.empty:
        return

    factor_columns = [
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
    ]

    print()
    print("=" * 80)
    print("СТАТИСТИКА FAMA–FRENCH 5")
    print("=" * 80)

    print(
        f"{'Фактор':<12}"
        f"{'Наблюдений':>15}"
        f"{'Средняя, %':>15}"
        f"{'Волатильность, %':>20}"
        f"{'Пропусков':>15}"
    )

    print(
        "-" * 80
    )

    for column in factor_columns:

        if column not in factors.columns:
            continue

        series = (
            factors[column]
            .dropna()
        )

        observations = len(
            series
        )

        missing = (
            len(factors)
            - observations
        )

        if observations > 0:

            mean_annual = (
                series.mean()
                * TRADING_DAYS
                * 100
            )

            volatility_annual = (
                series.std()
                * math.sqrt(TRADING_DAYS)
                * 100
            )

        else:

            mean_annual = None
            volatility_annual = None

        if mean_annual is None:

            mean_text = "—"

        else:

            mean_text = (
                f"{mean_annual:,.2f}"
                .replace(",", " ")
                .replace(".", ",")
            )

        if volatility_annual is None:

            volatility_text = "—"

        else:

            volatility_text = (
                f"{volatility_annual:,.2f}"
                .replace(",", " ")
                .replace(".", ",")
            )

        print(
            f"{column:<12}"
            f"{observations:>15}"
            f"{mean_text:>15}"
            f"{volatility_text:>20}"
            f"{missing:>15}"
        )


# ============================================================
# СТАТИСТИКА ASSIGNMENTS
# ============================================================

def print_assignment_statistics(
    assignments,
):

    if assignments.empty:
        return

    print()
    print("=" * 80)
    print("ПРОВЕРКА ФАКТОРНЫХ ПОРТФЕЛЕЙ")
    print("=" * 80)

    print(
        f"Всего assignments: "
        f"{len(assignments)}"
    )

    print(
        f"Компаний: "
        f"{assignments['ticker'].nunique()}"
    )

    print(
        f"Периодов: "
        f"{assignments['effective_date'].nunique()}"
    )

    print()

    for effective_date, group in (
        assignments
        .groupby("effective_date")
    ):

        report_year = int(
            group["report_year"]
            .iloc[0]
        )

        end_date = group[
            "end_date"
        ].iloc[0]

        size_counts = (
            group["size_group"]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        value_counts = (
            group["value_group"]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        profit_counts = (
            group["profitability_group"]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        investment_counts = (
            group["investment_group"]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        print(
            f"{effective_date.date()} → "
            f"{end_date} | "
            f"report_year={report_year}"
        )

        print(
            f"  SIZE: "
            f"S={size_counts.get('S', 0)}, "
            f"B={size_counts.get('B', 0)}"
        )

        print(
            f"  VALUE: "
            f"L={value_counts.get('L', 0)}, "
            f"N={value_counts.get('N', 0)}, "
            f"H={value_counts.get('H', 0)}, "
            f"NULL={value_counts.get(None, 0)}"
        )

        print(
            f"  PROFIT: "
            f"W={profit_counts.get('W', 0)}, "
            f"N={profit_counts.get('N', 0)}, "
            f"R={profit_counts.get('R', 0)}, "
            f"NULL={profit_counts.get(None, 0)}"
        )

        print(
            f"  INVEST: "
            f"A={investment_counts.get('A', 0)}, "
            f"N={investment_counts.get('N', 0)}, "
            f"C={investment_counts.get('C', 0)}, "
            f"NULL={investment_counts.get(None, 0)}"
        )


# ============================================================
# ПРОВЕРКА БД
# ============================================================

def print_db_counts(
    cursor,
):

    print()
    print("=" * 80)
    print("ПРОВЕРКА MART")
    print("=" * 80)

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM mart.ff5_assignments
        """
    )

    assignments_count = (
        cursor.fetchone()[0]
    )

    print(
        f"mart.ff5_assignments: "
        f"{assignments_count}"
    )

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM mart.ff5_factors
        """
    )

    factors_count = (
        cursor.fetchone()[0]
    )

    print(
        f"mart.ff5_factors: "
        f"{factors_count}"
    )

    cursor.execute(
        """
        SELECT
            MIN(trading_date),
            MAX(trading_date)
        FROM mart.ff5_factors
        """
    )

    min_date, max_date = (
        cursor.fetchone()
    )

    print(
        f"Период факторов: "
        f"{min_date} → {max_date}"
    )

    cursor.execute(
        """
        SELECT
            COUNT(*)
                FILTER (WHERE mkt_rf IS NULL),

            COUNT(*)
                FILTER (WHERE smb IS NULL),

            COUNT(*)
                FILTER (WHERE hml IS NULL),

            COUNT(*)
                FILTER (WHERE rmw IS NULL),

            COUNT(*)
                FILTER (WHERE cma IS NULL)

        FROM mart.ff5_factors
        """
    )

    (
        missing_mkt,
        missing_smb,
        missing_hml,
        missing_rmw,
        missing_cma,
    ) = cursor.fetchone()

    print()
    print(
        "Пропуски основных факторов:"
    )

    print(
        f"  MKT-RF: {missing_mkt}"
    )

    print(
        f"  SMB:    {missing_smb}"
    )

    print(
        f"  HML:    {missing_hml}"
    )

    print(
        f"  RMW:    {missing_rmw}"
    )

    print(
        f"  CMA:    {missing_cma}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ПОСТРОЕНИЕ FAMA–FRENCH 5 ФАКТОРОВ")
    print("=" * 80)

    connection = get_connection()

    try:

        # ====================================================
        # 1. СОЗДАНИЕ ТАБЛИЦ
        # ====================================================

        print()
        print(
            "[1] Создание таблиц"
        )

        with connection.cursor() as cursor:

            cursor.execute(
                CREATE_TABLES_SQL
            )

        connection.commit()

        # ====================================================
        # 2. ЗАГРУЗКА
        # ====================================================

        print()
        print(
            "[2] Загрузка данных из БД"
        )

        fundamentals = load_fundamentals(
            connection
        )

        prices = load_prices(
            connection
        )

        market_index = load_market_index(
            connection
        )

        risk_free = load_risk_free(
            connection
        )

        print(
            f"Fundamentals: "
            f"{len(fundamentals)}"
        )

        print(
            f"Цены акций: "
            f"{len(prices)}"
        )

        print(
            f"IMOEX: "
            f"{len(market_index)}"
        )

        print(
            f"ZCYC 1Y: "
            f"{len(risk_free)}"
        )

        if fundamentals.empty:

            raise ValueError(
                "core.fundamentals пустая"
            )

        if prices.empty:

            raise ValueError(
                "core.market_prices пустая"
            )

        if market_index.empty:

            raise ValueError(
                "core.market_indices для IMOEX пустая"
            )

        if risk_free.empty:

            raise ValueError(
                "core.risk_free_curve для 1Y пустая"
            )

        # ====================================================
        # 3. ПЕРИОДЫ
        # ====================================================

        max_trading_date = prices[
            "trading_date"
        ].max()

        periods = build_rebalance_periods(
            fundamentals,
            max_trading_date,
        )

        print()
        print(
            "[3] Периоды перебалансировки"
        )

        print(
            f"Периодов: "
            f"{len(periods)}"
        )

        for period in periods:

            print(
                f"  "
                f"{period['effective_date'].date()} "
                f"→ "
                f"{period['end_date'].date()} | "
                f"report_year="
                f"{period['report_year']}"
            )

        if not periods:

            raise ValueError(
                "Не удалось сформировать "
                "периоды перебалансировки"
            )

        # ====================================================
        # 4. ASSIGNMENTS
        # ====================================================

        print()
        print(
            "[4] Формирование факторных групп"
        )

        assignments = build_all_assignments(
            fundamentals,
            periods,
        )

        print(
            f"Assignments: "
            f"{len(assignments)}"
        )

        if assignments.empty:

            raise ValueError(
                "Факторные assignments не сформированы"
            )

        print_assignment_statistics(
            assignments
        )

        # ====================================================
        # 5. СОХРАНЕНИЕ ASSIGNMENTS
        # ====================================================

        print()
        print(
            "[5] Сохранение assignments"
        )

        saved_assignments = save_assignments(
            connection,
            assignments,
        )

        print(
            f"Сохранено assignments: "
            f"{saved_assignments}"
        )

        connection.commit()

        # ====================================================
        # 6. РАСЧЁТ FACTORS
        # ====================================================

        print()
        print(
            "[6] Расчёт факторов"
        )

        factors = build_all_factors(
            prices,
            market_index,
            risk_free,
            periods,
            assignments,
        )

        if factors.empty:

            raise ValueError(
                "Fama–French факторы не сформированы"
            )

        factors = clean_factors(
            factors
        )

        print(
            f"Дней факторов: "
            f"{len(factors)}"
        )

        print(
            f"Период: "
            f"{factors['trading_date'].min().date()} "
            f"→ "
            f"{factors['trading_date'].max().date()}"
        )

        # ====================================================
        # 7. СОХРАНЕНИЕ FACTORS
        # ====================================================

        print()
        print(
            "[7] Сохранение факторов"
        )

        saved_factors = save_factors(
            connection,
            factors,
        )

        print(
            f"Сохранено факторных наблюдений: "
            f"{saved_factors}"
        )

        connection.commit()

        # ====================================================
        # 8. СТАТИСТИКА
        # ====================================================

        print_factor_statistics(
            factors
        )

        # ====================================================
        # 9. ПРОВЕРКА БД
        # ====================================================

        with connection.cursor() as cursor:

            print_db_counts(
                cursor
            )

        connection.commit()

        print()
        print("=" * 80)
        print("FAMA–FRENCH 5 ФАКТОРЫ ПОСТРОЕНЫ")
        print("=" * 80)

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


if __name__ == "__main__":
    main()