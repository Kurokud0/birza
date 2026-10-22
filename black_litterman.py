import math

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from db import get_connection


# ============================================================
# ПАРАМЕТРЫ BLACK–LITTERMAN
# ============================================================

TRADING_DAYS = 252

# Коэффициент неприятия риска:
# pi = delta * Sigma * w_market
RISK_AVERSION = 2.5

# Параметр неопределённости prior
TAU = 0.05

# Окно для оценки ковариации
COVARIANCE_WINDOW = 252

# Минимум общих наблюдений
MIN_OBS = 100


# ============================================================
# SQL
# ============================================================

SQL_EXPECTED_RETURNS = """
SELECT
    ticker,
    rf_annual,
    expected_return_annual
FROM mart.expected_returns
WHERE ticker IS NOT NULL
ORDER BY ticker
"""


SQL_MARKET_CAPS = """
WITH ranked AS (
    SELECT
        f.ticker,
        f.report_year,
        f.period_end,
        f.market_cap,
        ROW_NUMBER() OVER (
            PARTITION BY f.ticker
            ORDER BY f.report_year DESC, f.period_end DESC
        ) AS rn
    FROM core.fundamentals f
    WHERE f.market_cap IS NOT NULL
      AND f.market_cap > 0
)
SELECT
    ticker,
    report_year,
    period_end,
    market_cap
FROM ranked
WHERE rn = 1
ORDER BY ticker
"""


SQL_PRICES = """
SELECT
    i.ticker,
    mp.trading_date,
    mp.close
FROM core.market_prices mp
JOIN core.instruments i
    ON i.instrument_id = mp.instrument_id
WHERE mp.close IS NOT NULL
ORDER BY mp.trading_date, i.ticker
"""


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def read_sql_df(sql, conn):
    """
    Чтение через cursor вместо pd.read_sql_query(),
    чтобы не получать предупреждение pandas о DBAPI.
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def to_float(value):
    if value is None:
        return np.nan

    try:
        result = float(value)

        if math.isfinite(result):
            return result

        return np.nan

    except (TypeError, ValueError):
        return np.nan


def to_python_float(value):
    """
    Явное преобразование numpy/pandas scalar в обычный Python float
    для корректной передачи в psycopg2.
    """
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def to_python_int(value):
    """
    Явное преобразование в обычный Python int.
    """
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return int(value)


def to_python_date(value):
    """
    Преобразование pandas Timestamp / numpy datetime
    в обычный Python date.
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date()

    if hasattr(value, "date") and not hasattr(value, "year"):
        return value.date()

    return value


def fmt_pct(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.{decimals}f}%"


# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================

def prepare_expected_returns(expected):
    if expected.empty:
        raise RuntimeError(
            "mart.expected_returns не содержит данных."
        )

    numeric_cols = [
        "rf_annual",
        "expected_return_annual",
    ]

    for col in numeric_cols:
        expected[col] = expected[col].map(to_float)

    expected["ticker"] = (
        expected["ticker"]
        .astype(str)
        .str.strip()
    )

    expected = expected.dropna(
        subset=[
            "ticker",
            "rf_annual",
            "expected_return_annual",
        ]
    )

    expected = (
        expected
        .drop_duplicates("ticker")
        .reset_index(drop=True)
    )

    if expected.empty:
        raise RuntimeError(
            "После очистки mart.expected_returns "
            "не осталось данных."
        )

    return expected


def prepare_market_caps(market_caps):
    if market_caps.empty:
        return market_caps

    market_caps["ticker"] = (
        market_caps["ticker"]
        .astype(str)
        .str.strip()
    )

    market_caps["market_cap"] = (
        market_caps["market_cap"]
        .map(to_float)
    )

    market_caps["report_year"] = pd.to_numeric(
        market_caps["report_year"],
        errors="coerce",
    )

    market_caps["period_end"] = pd.to_datetime(
        market_caps["period_end"],
        errors="coerce",
    )

    market_caps = market_caps.dropna(
        subset=[
            "ticker",
            "market_cap",
        ]
    )

    market_caps = market_caps[
        market_caps["market_cap"] > 0
    ]

    return market_caps


def prepare_prices(prices):
    if prices.empty:
        raise RuntimeError(
            "core.market_prices не содержит данных."
        )

    prices["ticker"] = (
        prices["ticker"]
        .astype(str)
        .str.strip()
    )

    prices["trading_date"] = pd.to_datetime(
        prices["trading_date"]
    )

    prices["close"] = prices["close"].map(
        to_float
    )

    prices = prices.dropna(
        subset=[
            "ticker",
            "trading_date",
            "close",
        ]
    )

    prices = prices[
        prices["close"] > 0
    ]

    return prices


# ============================================================
# РЫНОЧНЫЕ ВЕСА
# ============================================================

def build_market_weights(tickers, market_caps):
    """
    Для каждой акции берём последнюю доступную положительную
    market_cap.

    Если market_cap отсутствует, используется медианная
    капитализация остальных акций.
    """

    tickers = list(tickers)

    caps = (
        market_caps[
            market_caps["ticker"].isin(tickers)
        ]
        .sort_values(
            [
                "ticker",
                "report_year",
                "period_end",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=["ticker"],
            keep="first",
        )
        .copy()
    )

    available = caps["market_cap"].dropna()

    if available.empty:
        raise RuntimeError(
            "Нет ни одной положительной market_cap."
        )

    median_cap = float(
        available.median()
    )

    rows = []

    for ticker in tickers:
        match = caps[
            caps["ticker"] == ticker
        ]

        if not match.empty:
            row = match.iloc[0]

            report_year = (
                int(float(row["report_year"]))
                if pd.notna(row["report_year"])
                else None
            )

            period_end = (
                row["period_end"].date()
                if pd.notna(row["period_end"])
                else None
            )

            rows.append(
                {
                    "ticker": ticker,
                    "market_cap": float(
                        row["market_cap"]
                    ),
                    "market_cap_report_year": report_year,
                    "market_cap_period_end": period_end,
                    "weight_source": "MARKET_CAP",
                }
            )

        else:
            rows.append(
                {
                    "ticker": ticker,
                    "market_cap": median_cap,
                    "market_cap_report_year": None,
                    "market_cap_period_end": None,
                    "weight_source": "MEDIAN_FALLBACK",
                }
            )

    weights = pd.DataFrame(rows)

    total_cap = float(
        weights["market_cap"].sum()
    )

    if total_cap <= 0:
        raise RuntimeError(
            "Сумма рыночных капитализаций "
            "не положительна."
        )

    weights["market_weight"] = (
        weights["market_cap"]
        / total_cap
    )

    weights["market_weight"] = (
        weights["market_weight"]
        / weights["market_weight"].sum()
    )

    return weights


# ============================================================
# КОВАРИАЦИОННАЯ МАТРИЦА
# ============================================================

def build_covariance(prices, tickers):
    """
    Оценка covariance matrix:

    1. Дневные доходности.
    2. Только общие даты для всех акций.
    3. Последние 252 наблюдения.
    4. Ledoit-Wolf shrinkage.
    5. Перевод в годовую covariance matrix.
    """

    prices = prices[
        prices["ticker"].isin(tickers)
    ].copy()

    pivot = prices.pivot_table(
        index="trading_date",
        columns="ticker",
        values="close",
        aggfunc="last",
    )

    pivot = pivot.reindex(
        columns=tickers
    )

    returns = pivot.pct_change()

    complete = returns.dropna(
        how="any"
    )

    if len(complete) < MIN_OBS:
        raise RuntimeError(
            f"Недостаточно общих наблюдений: "
            f"{len(complete)}"
        )

    complete = complete.tail(
        COVARIANCE_WINDOW
    )

    if len(complete) < MIN_OBS:
        raise RuntimeError(
            f"После выбора окна осталось "
            f"{len(complete)} наблюдений."
        )

    daily_returns = complete.values

    estimator = LedoitWolf(
        assume_centered=False
    )

    estimator.fit(
        daily_returns
    )

    covariance_daily = (
        estimator.covariance_
    )

    covariance_annual = (
        covariance_daily
        * TRADING_DAYS
    )

    covariance_annual = pd.DataFrame(
        covariance_annual,
        index=tickers,
        columns=tickers,
    )

    return (
        covariance_annual,
        complete,
    )


# ============================================================
# BLACK–LITTERMAN
# ============================================================

def calculate_black_litterman(
    covariance,
    weights,
    expected_returns,
):
    """
    Black–Litterman с абсолютными FF5 views.

    Prior:

        pi = delta * Sigma * w

    Views:

        Q = E[R]_FF5 - RF

    P = I

    Omega:

        Omega = tau * diag(Sigma)

    Posterior:

        mu_BL =
            [(tau Sigma)^-1 + Omega^-1]^-1
            [(tau Sigma)^-1 pi + Omega^-1 Q]

    Всё рассчитывается для excess returns.
    """

    tickers = list(
        expected_returns["ticker"]
    )

    sigma = covariance.loc[
        tickers,
        tickers,
    ].values.astype(float)

    market_weights = (
        weights
        .set_index("ticker")
        .loc[
            tickers,
            "market_weight",
        ]
        .astype(float)
        .values
    )

    expected_df = (
        expected_returns
        .set_index("ticker")
        .loc[tickers]
    )

    # RF едина для всего универсума.
    rf_annual = float(
        expected_df["rf_annual"].median()
    )

    ff5_total = (
        expected_df[
            "expected_return_annual"
        ]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # FF5 views в excess return
    # --------------------------------------------------------

    q_excess = (
        ff5_total
        - rf_annual
    )

    # --------------------------------------------------------
    # Equilibrium prior
    # --------------------------------------------------------

    pi_excess = (
        RISK_AVERSION
        * sigma.dot(
            market_weights
        )
    )

    # --------------------------------------------------------
    # Views
    # --------------------------------------------------------

    n = len(tickers)

    tau_sigma = (
        TAU
        * sigma
    )

    # Omega = tau * diag(Sigma)
    omega_diag = (
        TAU
        * np.diag(sigma)
    )

    omega_diag = np.maximum(
        omega_diag,
        1e-12,
    )

    omega_inv = np.diag(
        1.0 / omega_diag
    )

    # --------------------------------------------------------
    # BL posterior
    # --------------------------------------------------------

    prior_precision = np.linalg.solve(
        tau_sigma,
        np.eye(n),
    )

    posterior_precision = (
        prior_precision
        + omega_inv
    )

    posterior_rhs = (
        prior_precision.dot(
            pi_excess
        )
        + omega_inv.dot(
            q_excess
        )
    )

    posterior_excess = np.linalg.solve(
        posterior_precision,
        posterior_rhs,
    )

    # Возвращаем RF.
    pi_total = (
        rf_annual
        + pi_excess
    )

    posterior_total = (
        rf_annual
        + posterior_excess
    )

    # --------------------------------------------------------
    # Вес views в изменении prior
    # --------------------------------------------------------

    prior_gap = (
        q_excess
        - pi_excess
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        view_weight = np.where(
            np.abs(prior_gap) > 1e-12,
            (
                posterior_excess
                - pi_excess
            )
            / prior_gap,
            0.0,
        )

    result = pd.DataFrame(
        {
            "ticker": tickers,
            "rf_annual": rf_annual,
            "market_weight": market_weights,
            "equilibrium_excess_return": pi_excess,
            "equilibrium_return": pi_total,
            "ff5_view_excess_return": q_excess,
            "ff5_view_return": ff5_total,
            "posterior_excess_return": posterior_excess,
            "posterior_return": posterior_total,
            "view_weight": view_weight,
        }
    )

    return result


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS mart.black_litterman;

CREATE TABLE mart.black_litterman (
    ticker VARCHAR(20) PRIMARY KEY,

    calc_date DATE NOT NULL,

    covariance_window INTEGER NOT NULL,
    covariance_observations INTEGER NOT NULL,

    risk_aversion DOUBLE PRECISION NOT NULL,
    tau DOUBLE PRECISION NOT NULL,

    rf_annual DOUBLE PRECISION NOT NULL,

    market_weight DOUBLE PRECISION NOT NULL,

    market_cap DOUBLE PRECISION NOT NULL,
    market_cap_report_year INTEGER,
    market_cap_period_end DATE,
    weight_source VARCHAR(30) NOT NULL,

    equilibrium_excess_return DOUBLE PRECISION NOT NULL,
    equilibrium_return DOUBLE PRECISION NOT NULL,

    ff5_view_excess_return DOUBLE PRECISION NOT NULL,
    ff5_view_return DOUBLE PRECISION NOT NULL,

    posterior_excess_return DOUBLE PRECISION NOT NULL,
    posterior_return DOUBLE PRECISION NOT NULL,

    view_weight DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def save_results(
    result,
    covariance_observations,
    calc_date,
):
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                CREATE_TABLE_SQL
            )

            insert_sql = """
            INSERT INTO mart.black_litterman (
                ticker,
                calc_date,
                covariance_window,
                covariance_observations,
                risk_aversion,
                tau,
                rf_annual,
                market_weight,
                market_cap,
                market_cap_report_year,
                market_cap_period_end,
                weight_source,
                equilibrium_excess_return,
                equilibrium_return,
                ff5_view_excess_return,
                ff5_view_return,
                posterior_excess_return,
                posterior_return,
                view_weight
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            """

            rows = []

            for _, row in result.iterrows():
                ticker = str(
                    row["ticker"]
                ).strip()

                calc_date_py = (
                    calc_date
                    if isinstance(
                        calc_date,
                        type(
                            pd.Timestamp(
                                "2020-01-01"
                            ).date()
                        ),
                    )
                    else to_python_date(
                        calc_date
                    )
                )

                market_cap_report_year = (
                    to_python_int(
                        row[
                            "market_cap_report_year"
                        ]
                    )
                )

                market_cap_period_end = (
                    to_python_date(
                        row[
                            "market_cap_period_end"
                        ]
                    )
                )

                rows.append(
                    (
                        ticker,
                        calc_date_py,
                        int(
                            COVARIANCE_WINDOW
                        ),
                        int(
                            covariance_observations
                        ),
                        float(
                            RISK_AVERSION
                        ),
                        float(
                            TAU
                        ),
                        to_python_float(
                            row["rf_annual"]
                        ),
                        to_python_float(
                            row["market_weight"]
                        ),
                        to_python_float(
                            row["market_cap"]
                        ),
                        market_cap_report_year,
                        market_cap_period_end,
                        str(
                            row["weight_source"]
                        ),
                        to_python_float(
                            row[
                                "equilibrium_excess_return"
                            ]
                        ),
                        to_python_float(
                            row[
                                "equilibrium_return"
                            ]
                        ),
                        to_python_float(
                            row[
                                "ff5_view_excess_return"
                            ]
                        ),
                        to_python_float(
                            row[
                                "ff5_view_return"
                            ]
                        ),
                        to_python_float(
                            row[
                                "posterior_excess_return"
                            ]
                        ),
                        to_python_float(
                            row[
                                "posterior_return"
                            ]
                        ),
                        to_python_float(
                            row["view_weight"]
                        ),
                    )
                )

            print(
                f"Подготовлено строк к сохранению: "
                f"{len(rows)}"
            )

            # ------------------------------------------------
            # Диагностика типов перед INSERT
            # ------------------------------------------------

            for row_index, row_data in enumerate(
                rows,
                start=1,
            ):
                integer_fields = {
                    "covariance_window": row_data[2],
                    "covariance_observations": row_data[3],
                    "market_cap_report_year": row_data[9],
                }

                for field_name, value in (
                    integer_fields.items()
                ):
                    if value is not None and not isinstance(
                        value,
                        int,
                    ):
                        raise TypeError(
                            f"Строка {row_index}, "
                            f"{field_name}: "
                            f"{value!r} "
                            f"не является Python int."
                        )

                    if value is not None:
                        if (
                            value < -2147483648
                            or value > 2147483647
                        ):
                            raise ValueError(
                                f"Строка {row_index}, "
                                f"{field_name}: "
                                f"{value} выходит "
                                f"за диапазон PostgreSQL INTEGER."
                            )

            cur.executemany(
                insert_sql,
                rows,
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("BLACK–LITTERMAN")
    print("=" * 100)
    print()

    print(
        "Загрузка FF5 expected returns, "
        "market caps и котировок..."
    )

    conn = get_connection()

    try:
        expected = read_sql_df(
            SQL_EXPECTED_RETURNS,
            conn,
        )

        market_caps = read_sql_df(
            SQL_MARKET_CAPS,
            conn,
        )

        prices = read_sql_df(
            SQL_PRICES,
            conn,
        )

    finally:
        conn.close()

    expected = prepare_expected_returns(
        expected
    )

    market_caps = prepare_market_caps(
        market_caps
    )

    prices = prepare_prices(
        prices
    )

    tickers = expected[
        "ticker"
    ].tolist()

    print(
        f"Акций в универсе: "
        f"{len(tickers)}"
    )

    print(
        "Тикеры: "
        + ", ".join(tickers)
    )

    # ========================================================
    # РЫНОЧНЫЕ ВЕСА
    # ========================================================

    weights = build_market_weights(
        tickers,
        market_caps,
    )

    fallback_count = int(
        (
            weights["weight_source"]
            == "MEDIAN_FALLBACK"
        ).sum()
    )

    print()
    print(
        "Рыночные веса построены."
    )

    print(
        f"Из market_cap: "
        f"{len(tickers) - fallback_count}"
    )

    print(
        f"Median fallback: "
        f"{fallback_count}"
    )

    # ========================================================
    # КОВАРИАЦИЯ
    # ========================================================

    print()
    print(
        "Расчёт ковариационной матрицы "
        "с Ledoit–Wolf..."
    )

    covariance, common_returns = (
        build_covariance(
            prices,
            tickers,
        )
    )

    covariance_start = (
        common_returns.index.min()
    )

    covariance_end = (
        common_returns.index.max()
    )

    print(
        "Окно ковариации: "
        f"{covariance_start.date()} → "
        f"{covariance_end.date()}"
    )

    print(
        f"Общих наблюдений: "
        f"{len(common_returns)}"
    )

    # ========================================================
    # BLACK–LITTERMAN
    # ========================================================

    result = calculate_black_litterman(
        covariance,
        weights,
        expected,
    )

    result = result.merge(
        weights[
            [
                "ticker",
                "market_cap",
                "market_cap_report_year",
                "market_cap_period_end",
                "weight_source",
            ]
        ],
        on="ticker",
        how="left",
    )

    result = result.sort_values(
        "posterior_return",
        ascending=False,
    ).reset_index(
        drop=True
    )

    calc_date = (
        covariance_end.date()
    )

    # ========================================================
    # ВЫВОД
    # ========================================================

    print()
    print("=" * 100)
    print("BLACK–LITTERMAN POSTERIOR RETURNS")
    print("=" * 100)

    print(
        f"Дата расчёта: "
        f"{calc_date}"
    )

    print(
        f"Risk aversion δ: "
        f"{RISK_AVERSION}"
    )

    print(
        f"Tau: "
        f"{TAU}"
    )

    print(
        f"RF: "
        f"{fmt_pct(result['rf_annual'].iloc[0])}"
    )

    print()
    print(
        f"{'Ticker':<8}"
        f"{'Mkt W':>9}"
        f"{'Prior':>11}"
        f"{'FF5 View':>11}"
        f"{'Posterior':>12}"
        f"{'View W':>9}"
    )

    print("-" * 75)

    for _, row in result.iterrows():
        print(
            f"{row['ticker']:<8}"
            f"{fmt_pct(row['market_weight']):>9}"
            f"{fmt_pct(row['equilibrium_return']):>11}"
            f"{fmt_pct(row['ff5_view_return']):>11}"
            f"{fmt_pct(row['posterior_return']):>12}"
            f"{float(row['view_weight']):>9.3f}"
        )

    # ========================================================
    # ПРОВЕРКИ
    # ========================================================

    if not np.isfinite(
        result["posterior_return"].astype(float)
    ).all():
        raise RuntimeError(
            "В posterior_return обнаружены "
            "NaN или бесконечные значения."
        )

    weight_sum = float(
        result["market_weight"]
        .astype(float)
        .sum()
    )

    if not np.isclose(
        weight_sum,
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Сумма рыночных весов не равна 1."
        )

    print()
    print(
        "Контроль:"
    )

    print(
        f"Сумма market weights: "
        f"{weight_sum:.6f}"
    )

    print(
        f"Posterior NaN: "
        f"{result['posterior_return'].isna().sum()}"
    )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    save_results(
        result=result,
        covariance_observations=len(
            common_returns
        ),
        calc_date=calc_date,
    )

    print()
    print("=" * 100)
    print("Сохранение в БД: OK")
    print(
        "Таблица: "
        "mart.black_litterman"
    )
    print(
        f"Сохранено акций: "
        f"{len(result)}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()