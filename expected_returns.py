import math
import numpy as np
import pandas as pd

from db import get_connection


TRADING_DAYS = 252
MIN_OBS = 100


CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS mart.expected_returns;

CREATE TABLE mart.expected_returns (
    ticker VARCHAR(20) PRIMARY KEY,

    calc_date DATE NOT NULL,
    factor_start_date DATE NOT NULL,
    factor_end_date DATE NOT NULL,
    factor_observations INTEGER NOT NULL,
    beta_observations INTEGER NOT NULL,

    rf_annual DOUBLE PRECISION NOT NULL,

    mkt_premium_annual DOUBLE PRECISION NOT NULL,
    smb_premium_annual DOUBLE PRECISION NOT NULL,
    hml_premium_annual DOUBLE PRECISION NOT NULL,
    rmw_premium_annual DOUBLE PRECISION NOT NULL,
    cma_premium_annual DOUBLE PRECISION NOT NULL,

    alpha_annual DOUBLE PRECISION,
    alpha_pvalue DOUBLE PRECISION,

    beta_mkt DOUBLE PRECISION NOT NULL,
    beta_smb DOUBLE PRECISION NOT NULL,
    beta_hml DOUBLE PRECISION NOT NULL,
    beta_rmw DOUBLE PRECISION NOT NULL,
    beta_cma DOUBLE PRECISION NOT NULL,

    factor_contribution_mkt DOUBLE PRECISION NOT NULL,
    factor_contribution_smb DOUBLE PRECISION NOT NULL,
    factor_contribution_hml DOUBLE PRECISION NOT NULL,
    factor_contribution_rmw DOUBLE PRECISION NOT NULL,
    factor_contribution_cma DOUBLE PRECISION NOT NULL,

    expected_excess_return_annual DOUBLE PRECISION NOT NULL,
    expected_return_annual DOUBLE PRECISION NOT NULL,

    r_squared DOUBLE PRECISION,
    adj_r_squared DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


SQL_FACTORS = """
SELECT
    trading_date,
    mkt_rf,
    smb,
    hml,
    rmw,
    cma,
    rf
FROM mart.ff5_factors
WHERE trading_date IS NOT NULL
ORDER BY trading_date
"""


SQL_REGRESSION = """
SELECT
    ticker,
    alpha_annualized,
    p_alpha,
    beta_mkt,
    beta_smb,
    beta_hml,
    beta_rmw,
    beta_cma,
    r_squared,
    adj_r_squared,
    n_obs
FROM mart.ff5_regression
ORDER BY ticker
"""


def read_sql_df(sql, conn):
    """Читаем через cursor, чтобы не получать предупреждение pandas."""
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
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fmt_pct(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def prepare_factors(factors):
    if factors.empty:
        raise RuntimeError("mart.ff5_factors не содержит данных.")

    factors["trading_date"] = pd.to_datetime(factors["trading_date"]).dt.date

    numeric_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    for col in numeric_cols:
        factors[col] = factors[col].map(to_float)

    factors = factors.sort_values("trading_date").drop_duplicates(
        subset=["trading_date"], keep="last"
    )
    return factors


def prepare_regression(regression):
    if regression.empty:
        raise RuntimeError("mart.ff5_regression не содержит результатов регрессии.")

    numeric_cols = [
        "alpha_annualized", "p_alpha", "beta_mkt", "beta_smb",
        "beta_hml", "beta_rmw", "beta_cma", "r_squared",
        "adj_r_squared", "n_obs",
    ]
    for col in numeric_cols:
        regression[col] = regression[col].map(to_float)
    return regression


def calculate_factor_premiums(factors):
    """
    Оцениваем годовые премии факторов по всей доступной истории,
    на которой одновременно присутствуют все пять FF5-факторов.

    Премия = средняя дневная доходность фактора * 252.

    В отличие от предыдущей версии здесь НЕ используется только
    последнее окно из 252 дней. Это снижает чувствительность
    ожидаемой доходности к краткосрочному рыночному периоду.
    """
    factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma"]

    history = (
        factors
        .dropna(subset=factor_cols)
        .sort_values("trading_date")
        .reset_index(drop=True)
    )

    if len(history) < MIN_OBS:
        raise RuntimeError(
            f"Недостаточно полных наблюдений факторов: "
            f"{len(history)}, минимум {MIN_OBS}."
        )

    premiums = history[factor_cols].mean() * TRADING_DAYS

    return history, premiums


def calculate_current_rf(factors):
    """Переводим последнюю дневную RF в годовую ставку сложного процента."""
    rf_series = factors.dropna(subset=["rf"])
    if rf_series.empty:
        raise RuntimeError("В mart.ff5_factors нет доступных значений RF.")

    latest_rf_daily = float(rf_series.iloc[-1]["rf"])
    if latest_rf_daily <= -1:
        raise RuntimeError(f"Некорректная дневная RF: {latest_rf_daily}")

    return (1.0 + latest_rf_daily) ** TRADING_DAYS - 1.0


def calculate_expected_returns(regression, factors):
    """
    E[R_i] = R_f + βMKT*E[MKT-RF] + βSMB*E[SMB]
                  + βHML*E[HML] + βRMW*E[RMW] + βCMA*E[CMA] + alpha.

    Альфа включается только при p < 0.05.
    """
    window, premiums = calculate_factor_premiums(factors)
    rf_annual = calculate_current_rf(factors)

    premium_map = {
        "mkt_rf": float(premiums["mkt_rf"]),
        "smb": float(premiums["smb"]),
        "hml": float(premiums["hml"]),
        "rmw": float(premiums["rmw"]),
        "cma": float(premiums["cma"]),
    }

    results = []

    for _, row in regression.iterrows():
        ticker = str(row["ticker"]).strip()

        betas = {
            "mkt_rf": to_float(row["beta_mkt"]),
            "smb": to_float(row["beta_smb"]),
            "hml": to_float(row["beta_hml"]),
            "rmw": to_float(row["beta_rmw"]),
            "cma": to_float(row["beta_cma"]),
        }

        if any(pd.isna(v) for v in betas.values()):
            print(f"{ticker:<7} SKIP | отсутствует один из β")
            continue

        p_alpha = to_float(row["p_alpha"])
        alpha_annualized = to_float(row["alpha_annualized"])

        alpha_used = (
            alpha_annualized
            if pd.notna(alpha_annualized) and pd.notna(p_alpha) and p_alpha < 0.05
            else 0.0
        )

        contribution_mkt = betas["mkt_rf"] * premium_map["mkt_rf"]
        contribution_smb = betas["smb"] * premium_map["smb"]
        contribution_hml = betas["hml"] * premium_map["hml"]
        contribution_rmw = betas["rmw"] * premium_map["rmw"]
        contribution_cma = betas["cma"] * premium_map["cma"]

        expected_excess = (
            alpha_used
            + contribution_mkt
            + contribution_smb
            + contribution_hml
            + contribution_rmw
            + contribution_cma
        )
        expected_return = rf_annual + expected_excess

        results.append({
            "ticker": ticker,
            "calc_date": window["trading_date"].max(),
            "factor_start_date": window["trading_date"].min(),
            "factor_end_date": window["trading_date"].max(),
            "factor_observations": len(window),
            "beta_observations": int(row["n_obs"]) if pd.notna(row["n_obs"]) else 0,
            "rf_annual": rf_annual,
            "mkt_premium_annual": premium_map["mkt_rf"],
            "smb_premium_annual": premium_map["smb"],
            "hml_premium_annual": premium_map["hml"],
            "rmw_premium_annual": premium_map["rmw"],
            "cma_premium_annual": premium_map["cma"],
            "alpha_annual": alpha_annualized if pd.notna(alpha_annualized) else None,
            "alpha_pvalue": p_alpha if pd.notna(p_alpha) else None,
            "beta_mkt": betas["mkt_rf"],
            "beta_smb": betas["smb"],
            "beta_hml": betas["hml"],
            "beta_rmw": betas["rmw"],
            "beta_cma": betas["cma"],
            "factor_contribution_mkt": contribution_mkt,
            "factor_contribution_smb": contribution_smb,
            "factor_contribution_hml": contribution_hml,
            "factor_contribution_rmw": contribution_rmw,
            "factor_contribution_cma": contribution_cma,
            "expected_excess_return_annual": expected_excess,
            "expected_return_annual": expected_return,
            "r_squared": row["r_squared"] if pd.notna(row["r_squared"]) else None,
            "adj_r_squared": row["adj_r_squared"] if pd.notna(row["adj_r_squared"]) else None,
        })

    result = pd.DataFrame(results)
    if result.empty:
        raise RuntimeError("Ни одна акция не получила ожидаемую доходность.")

    return result, window, rf_annual, premium_map


def save_results(result):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)

            insert_sql = """
            INSERT INTO mart.expected_returns (
                ticker, calc_date, factor_start_date, factor_end_date,
                factor_observations, beta_observations, rf_annual, mkt_premium_annual,
                smb_premium_annual, hml_premium_annual, rmw_premium_annual,
                cma_premium_annual, alpha_annual, alpha_pvalue,
                beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma,
                factor_contribution_mkt, factor_contribution_smb,
                factor_contribution_hml, factor_contribution_rmw,
                factor_contribution_cma, expected_excess_return_annual,
                expected_return_annual, r_squared, adj_r_squared
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """

            rows = []
            for _, row in result.iterrows():
                rows.append((
                    row["ticker"], row["calc_date"],
                    row["factor_start_date"], row["factor_end_date"],
                    int(row["factor_observations"]), int(row["beta_observations"]),
                    row["rf_annual"], row["mkt_premium_annual"],
                    row["smb_premium_annual"], row["hml_premium_annual"],
                    row["rmw_premium_annual"], row["cma_premium_annual"],
                    row["alpha_annual"], row["alpha_pvalue"], row["beta_mkt"],
                    row["beta_smb"], row["beta_hml"], row["beta_rmw"], row["beta_cma"],
                    row["factor_contribution_mkt"], row["factor_contribution_smb"],
                    row["factor_contribution_hml"], row["factor_contribution_rmw"],
                    row["factor_contribution_cma"], row["expected_excess_return_annual"],
                    row["expected_return_annual"], row["r_squared"], row["adj_r_squared"],
                ))

            cur.executemany(insert_sql, rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    print("=" * 100)
    print("РАСЧЁТ ОЖИДАЕМОЙ ДОХОДНОСТИ ПО FF5")
    print("=" * 100)
    print()

    print("Загрузка FF5-факторов и результатов регрессии...")

    conn = get_connection()
    try:
        factors = read_sql_df(SQL_FACTORS, conn)
        regression = read_sql_df(SQL_REGRESSION, conn)
    finally:
        conn.close()

    factors = prepare_factors(factors)
    regression = prepare_regression(regression)

    print(f"Факторов загружено: {len(factors):,}".replace(",", " "))
    print(f"Регрессий загружено: {len(regression):,}".replace(",", " "))
    print(
        "Период факторов: "
        f"{factors['trading_date'].min()} → {factors['trading_date'].max()}"
    )

    result, window, rf_annual, premium_map = calculate_expected_returns(
        regression, factors
    )

    print()
    print(
        f"Исторический период оценки премий: "
        f"{window['trading_date'].min()} → "
        f"{window['trading_date'].max()}"
    )
    print(
        f"Полных факторных наблюдений: "
        f"{len(window):,}".replace(",", " ")
    )
    print(f"Текущая RF годовая: {fmt_pct(rf_annual)}")

    print()
    print("Исторические годовые факторные премии:")
    print(f"  MKT-RF: {fmt_pct(premium_map['mkt_rf'])}")
    print(f"  SMB:    {fmt_pct(premium_map['smb'])}")
    print(f"  HML:    {fmt_pct(premium_map['hml'])}")
    print(f"  RMW:    {fmt_pct(premium_map['rmw'])}")
    print(f"  CMA:    {fmt_pct(premium_map['cma'])}")

    print()
    print("=" * 100)
    print("ОЖИДАЕМАЯ ДОХОДНОСТЬ АКЦИЙ")
    print("=" * 100)

    print(
        f"{'Ticker':<8}{'RF':>10}{'MKT':>10}{'SMB':>10}{'HML':>10}"
        f"{'RMW':>10}{'CMA':>10}{'E[R]':>11}"
    )
    print("-" * 100)

    result = result.sort_values("expected_return_annual", ascending=False)

    for _, row in result.iterrows():
        print(
            f"{row['ticker']:<8}"
            f"{fmt_pct(row['rf_annual']):>10}"
            f"{fmt_pct(row['factor_contribution_mkt']):>10}"
            f"{fmt_pct(row['factor_contribution_smb']):>10}"
            f"{fmt_pct(row['factor_contribution_hml']):>10}"
            f"{fmt_pct(row['factor_contribution_rmw']):>10}"
            f"{fmt_pct(row['factor_contribution_cma']):>10}"
            f"{fmt_pct(row['expected_return_annual']):>11}"
        )

    save_results(result)

    print()
    print("=" * 100)
    print("Сохранение в БД: OK")
    print("Таблица: mart.expected_returns")
    print(f"Сохранено акций: {len(result)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
