import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from db import get_connection


MIN_OBSERVATIONS = 60
HAC_LAGS = 5
FACTOR_TABLE = "mart.ff5_factors"
RESULT_TABLE = "mart.ff5_regression"



def get_columns(cur, schema: str, table: str) -> list[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return [row[0] for row in cur.fetchall()]



def pick_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    by_lower = {column.lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]

    raise RuntimeError(
        f"Не найден столбец для '{label}'. "
        f"Ожидался один из: {', '.join(candidates)}. "
        f"Фактические столбцы: {', '.join(columns)}"
    )



def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'



def load_data(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        factor_columns = get_columns(cur, "mart", "ff5_factors")

    date_col = pick_column(
        factor_columns,
        ["trading_date", "factor_date", "date", "dt"],
        "даты факторов",
    )
    rf_col = pick_column(
        factor_columns,
        ["rf", "risk_free", "risk_free_rate"],
        "безрисковой ставки",
    )
    mkt_col = pick_column(
        factor_columns,
        ["mkt_rf", "MKT-RF", "mkt-rf", "market_rf"],
        "MKT-RF",
    )
    smb_col = pick_column(
        factor_columns,
        ["smb", "SMB"],
        "SMB",
    )
    hml_col = pick_column(
        factor_columns,
        ["hml", "HML"],
        "HML",
    )
    rmw_col = pick_column(
        factor_columns,
        ["rmw", "RMW"],
        "RMW",
    )
    cma_col = pick_column(
        factor_columns,
        ["cma", "CMA"],
        "CMA",
    )

    sql = f"""
        WITH stock_prices AS (
            SELECT
                i.ticker,
                mp.trading_date,
                mp.close,
                LAG(mp.close) OVER (
                    PARTITION BY i.ticker
                    ORDER BY mp.trading_date
                ) AS prev_close
            FROM core.market_prices mp
            JOIN core.instruments i
              ON i.instrument_id = mp.instrument_id
            WHERE mp.close IS NOT NULL
              AND mp.close > 0
        )
        SELECT
            s.ticker,
            s.trading_date,
            (s.close / NULLIF(s.prev_close, 0) - 1.0) AS stock_return,
            f.{quote_ident(rf_col)} AS rf,
            f.{quote_ident(mkt_col)} AS mkt_rf,
            f.{quote_ident(smb_col)} AS smb,
            f.{quote_ident(hml_col)} AS hml,
            f.{quote_ident(rmw_col)} AS rmw,
            f.{quote_ident(cma_col)} AS cma
        FROM stock_prices s
        JOIN {quote_ident('mart')}.{quote_ident('ff5_factors')} f
          ON f.{quote_ident(date_col)} = s.trading_date
        WHERE s.prev_close IS NOT NULL
        ORDER BY s.ticker, s.trading_date
    """

    return pd.read_sql_query(sql, conn)



def to_float_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result



def create_result_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {RESULT_TABLE}")
        cur.execute(
            f"""
            CREATE TABLE {RESULT_TABLE} (
                ticker VARCHAR(20) PRIMARY KEY,
                alpha_daily DOUBLE PRECISION,
                alpha_annualized DOUBLE PRECISION,
                beta_mkt DOUBLE PRECISION,
                beta_smb DOUBLE PRECISION,
                beta_hml DOUBLE PRECISION,
                beta_rmw DOUBLE PRECISION,
                beta_cma DOUBLE PRECISION,
                p_alpha DOUBLE PRECISION,
                p_beta_mkt DOUBLE PRECISION,
                p_beta_smb DOUBLE PRECISION,
                p_beta_hml DOUBLE PRECISION,
                p_beta_rmw DOUBLE PRECISION,
                p_beta_cma DOUBLE PRECISION,
                r_squared DOUBLE PRECISION,
                adj_r_squared DOUBLE PRECISION,
                n_obs INTEGER,
                start_date DATE,
                end_date DATE,
                hac_lags INTEGER NOT NULL DEFAULT {HAC_LAGS},
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    conn.commit()



def safe_float(value) -> float | None:
    if value is None:
        return None

    value = float(value)
    return value if math.isfinite(value) else None



def run_regression(ticker_df: pd.DataFrame, ticker: str) -> Dict:
    columns = ["stock_return", "rf", "mkt_rf", "smb", "hml", "rmw", "cma"]

    data = ticker_df[["trading_date", *columns]].copy()
    data = to_float_columns(data, columns)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    data = data.sort_values("trading_date")

    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Недостаточно наблюдений: {len(data)} < {MIN_OBSERVATIONS}"
        )

    data["excess_return"] = data["stock_return"] - data["rf"]

    X = data[["mkt_rf", "smb", "hml", "rmw", "cma"]]
    X = sm.add_constant(X, has_constant="add")
    y = data["excess_return"]

    model = sm.OLS(y, X, missing="drop")
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

    params = results.params
    pvalues = results.pvalues

    alpha_daily = safe_float(params.get("const"))
    alpha_annualized = (
        safe_float((math.exp(alpha_daily) - 1.0) * 252.0)
        if alpha_daily is not None
        else None
    )

    # Для небольших ежедневных доходностей более привычная в отчётах
    # интерпретация — приблизительно alpha_daily * 252.
    if alpha_daily is not None:
        alpha_annualized = safe_float(alpha_daily * 252.0)

    return {
        "ticker": ticker,
        "alpha_daily": alpha_daily,
        "alpha_annualized": alpha_annualized,
        "beta_mkt": safe_float(params.get("mkt_rf")),
        "beta_smb": safe_float(params.get("smb")),
        "beta_hml": safe_float(params.get("hml")),
        "beta_rmw": safe_float(params.get("rmw")),
        "beta_cma": safe_float(params.get("cma")),
        "p_alpha": safe_float(pvalues.get("const")),
        "p_beta_mkt": safe_float(pvalues.get("mkt_rf")),
        "p_beta_smb": safe_float(pvalues.get("smb")),
        "p_beta_hml": safe_float(pvalues.get("hml")),
        "p_beta_rmw": safe_float(pvalues.get("rmw")),
        "p_beta_cma": safe_float(pvalues.get("cma")),
        "r_squared": safe_float(results.rsquared),
        "adj_r_squared": safe_float(results.rsquared_adj),
        "n_obs": int(results.nobs),
        "start_date": data["trading_date"].min(),
        "end_date": data["trading_date"].max(),
        "hac_lags": HAC_LAGS,
    }



def save_results(conn, results: list[Dict]) -> None:
    if not results:
        raise RuntimeError("Нет результатов для сохранения.")

    sql = f"""
        INSERT INTO {RESULT_TABLE} (
            ticker,
            alpha_daily,
            alpha_annualized,
            beta_mkt,
            beta_smb,
            beta_hml,
            beta_rmw,
            beta_cma,
            p_alpha,
            p_beta_mkt,
            p_beta_smb,
            p_beta_hml,
            p_beta_rmw,
            p_beta_cma,
            r_squared,
            adj_r_squared,
            n_obs,
            start_date,
            end_date,
            hac_lags
        )
        VALUES (
            %(ticker)s,
            %(alpha_daily)s,
            %(alpha_annualized)s,
            %(beta_mkt)s,
            %(beta_smb)s,
            %(beta_hml)s,
            %(beta_rmw)s,
            %(beta_cma)s,
            %(p_alpha)s,
            %(p_beta_mkt)s,
            %(p_beta_smb)s,
            %(p_beta_hml)s,
            %(p_beta_rmw)s,
            %(p_beta_cma)s,
            %(r_squared)s,
            %(adj_r_squared)s,
            %(n_obs)s,
            %(start_date)s,
            %(end_date)s,
            %(hac_lags)s
        )
    """

    with conn.cursor() as cur:
        cur.executemany(sql, results)

    conn.commit()



def format_number(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{decimals}f}"



def format_percent(value, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"



def print_summary(results: list[Dict]) -> None:
    result_df = pd.DataFrame(results).sort_values("ticker")

    print("\n" + "=" * 100)
    print("FF5 ФАКТОРНЫЙ АНАЛИЗ")
    print("=" * 100)
    print(f"Акций: {len(result_df)}")
    print(
        f"Период регрессии: {result_df['start_date'].min()} → "
        f"{result_df['end_date'].max()}"
    )
    print(f"HAC lags: {HAC_LAGS}")
    print("\nРезультаты:")
    print("-" * 100)
    print(
        f"{'Ticker':<8} "
        f"{'Alpha год.':>12} "
        f"{'Beta MKT':>10} "
        f"{'Beta SMB':>10} "
        f"{'Beta HML':>10} "
        f"{'Beta RMW':>10} "
        f"{'Beta CMA':>10} "
        f"{'R²':>8} "
        f"{'Obs':>6}"
    )
    print("-" * 100)

    for row in result_df.to_dict("records"):
        print(
            f"{row['ticker']:<8} "
            f"{format_percent(row['alpha_annualized']):>12} "
            f"{format_number(row['beta_mkt'], 3):>10} "
            f"{format_number(row['beta_smb'], 3):>10} "
            f"{format_number(row['beta_hml'], 3):>10} "
            f"{format_number(row['beta_rmw'], 3):>10} "
            f"{format_number(row['beta_cma'], 3):>10} "
            f"{format_number(row['r_squared'], 3):>8} "
            f"{row['n_obs']:>6}"
        )

    print("-" * 100)

    print("\nP-value коэффициентов:")
    print("alpha / MKT / SMB / HML / RMW / CMA")
    for row in result_df.to_dict("records"):
        print(
            f"{row['ticker']:<8} "
            f"{format_number(row['p_alpha'], 4):>8} "
            f"{format_number(row['p_beta_mkt'], 4):>8} "
            f"{format_number(row['p_beta_smb'], 4):>8} "
            f"{format_number(row['p_beta_hml'], 4):>8} "
            f"{format_number(row['p_beta_rmw'], 4):>8} "
            f"{format_number(row['p_beta_cma'], 4):>8}"
        )



def main() -> None:
    print("Загрузка данных для FF5-регрессии...")

    conn = get_connection()

    try:
        df = load_data(conn)

        if df.empty:
            raise RuntimeError("Запрос не вернул данных для регрессии.")

        print(f"Загружено строк: {len(df):,}".replace(",", " "))
        print(f"Акций: {df['ticker'].nunique()}")
        print(f"Период данных: {df['trading_date'].min()} → {df['trading_date'].max()}")

        result_rows: list[Dict] = []
        failed: list[tuple[str, str]] = []

        for ticker, ticker_df in df.groupby("ticker", sort=True):
            try:
                result = run_regression(ticker_df, ticker)
                result_rows.append(result)

                print(
                    f"{ticker:<8} OK | "
                    f"n={result['n_obs']} | "
                    f"period={result['start_date']}→{result['end_date']} | "
                    f"R²={format_number(result['r_squared'], 3)}"
                )
            except Exception as exc:
                failed.append((ticker, str(exc)))
                print(f"{ticker:<8} ERROR | {exc}")

        if not result_rows:
            raise RuntimeError("Ни одна акция не прошла регрессию.")

        create_result_table(conn)
        save_results(conn, result_rows)
        print_summary(result_rows)

        print("\nСохранение в БД: OK")
        print(f"Таблица: {RESULT_TABLE}")
        print(f"Сохранено регрессий: {len(result_rows)}")

        if failed:
            print("\nАкции с ошибками:")
            for ticker, reason in failed:
                print(f"- {ticker}: {reason}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
