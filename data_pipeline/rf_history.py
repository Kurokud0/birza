"""Историческая бескупонная доходность ОФЗ на год по данным ЦБ"""

from __future__ import annotations

import calendar
import re
import sys
import time
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from portfolio_app.db import get_connection
except Exception as exc:
    print("Не удалось импортировать get_connection из portfolio_app.db.")
    print(f"Причина: {type(exc).__name__}: {exc}")
    sys.exit(1)

CBR_URL = "https://www.cbr.ru/hd_base/zcyc_params/"
TRADING_DAYS = 252
REQUEST_TIMEOUT = 30
PAUSE_SECONDS = 0.12

SQL_DATE_BOUNDS = """
SELECT MIN(trading_date)::date, MAX(trading_date)::date
FROM mart.market_daily
WHERE trading_date IS NOT NULL AND close > 0
"""
SQL_FF5 = """
SELECT trading_date, rf
FROM mart.ff5_factors
WHERE trading_date IS NOT NULL AND rf IS NOT NULL
ORDER BY trading_date
"""
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS mart.rf_zcyc_history (
    trading_date DATE PRIMARY KEY,
    yield_1y_pct DOUBLE PRECISION NOT NULL,
    rf_daily DOUBLE PRECISION NOT NULL,
    source_name VARCHAR(80) NOT NULL DEFAULT 'CBR_ZCYC_1Y',
    downloaded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (yield_1y_pct > -100),
    CHECK (rf_daily > -1)
)
"""
UPSERT_SQL = """
INSERT INTO mart.rf_zcyc_history
    (trading_date, yield_1y_pct, rf_daily, source_name, downloaded_at)
VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
ON CONFLICT (trading_date) DO UPDATE SET
    yield_1y_pct = EXCLUDED.yield_1y_pct,
    rf_daily = EXCLUDED.rf_daily,
    source_name = EXCLUDED.source_name,
    downloaded_at = CURRENT_TIMESTAMP
"""


class TableParser(HTMLParser):
    """Простой разбор HTML-таблиц средствами стандартной библиотеки."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self.table_depth = 0
        self.active_table: int | None = None
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            if self.table_depth == 0:
                self.tables.append([])
                self.active_table = len(self.tables) - 1
            self.table_depth += 1
            return
        if self.table_depth != 1 or self.active_table is None:
            return
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th") and self.current_row is not None:
            self.current_cell = []

    def handle_data(self, data):
        if self.table_depth == 1 and self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self.table_depth == 1:
            if self.current_cell is not None and self.current_row is not None:
                text = " ".join("".join(self.current_cell).split())
                self.current_row.append(text)
            self.current_cell = None
        elif tag == "tr" and self.table_depth == 1:
            if self.current_row is not None and self.active_table is not None:
                self.tables[self.active_table].append(self.current_row)
            self.current_row = None
            self.current_cell = None
        elif tag == "table":
            self.table_depth = max(0, self.table_depth - 1)
            if self.table_depth == 0:
                self.active_table = None
                self.current_row = None
                self.current_cell = None


def parse_number(text: str) -> float | None:
    s = (
        text.replace("\xa0", " ")
        .strip()
        .replace(" ", "")
        .replace(",", ".")
        .replace("−", "-")
    )
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", s):
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def parse_cbr_html(html: str) -> pd.DataFrame:
    parser = TableParser()
    parser.feed(html)
    rows_out = []
    date_pattern = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
    for table in parser.tables:
        for cells in table:
            row_date = None
            date_cell_index = None
            for idx, cell in enumerate(cells):
                match = date_pattern.search(cell)
                if match:
                    try:
                        row_date = datetime.strptime(match.group(0), "%d.%m.%Y").date()
                        date_cell_index = idx
                        break
                    except ValueError:
                        pass
            if row_date is None:
                continue
            nums = []
            for idx, cell in enumerate(cells):
                if idx == date_cell_index:
                    continue
                number = parse_number(cell)
                if number is not None:
                    nums.append(number)
            # Сроки на официальной странице идут в порядке 0.25, 0.5, 0.75, 1, 2...
            # Нужен четвёртый столбец доходностей: один год. Достаточно четырёх столбцов,
            # даже если ЦБ уберёт из таблицы наиболее длинный срок.
            if len(nums) >= 4:
                y1 = float(nums[3])
                if y1 > -100:
                    rows_out.append((row_date, y1))
    if not rows_out:
        return pd.DataFrame(columns=["trading_date", "yield_1y_pct"])
    out = pd.DataFrame(rows_out, columns=["trading_date", "yield_1y_pct"])
    out = (
        out.drop_duplicates("trading_date", keep="last")
        .sort_values("trading_date")
        .reset_index(drop=True)
    )
    return out


def month_chunks(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        chunk_end = min(date(current.year, current.month, last_day), end)
        chunk_start = max(current, start)
        yield chunk_start, chunk_end
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def get_page(session: requests.Session, start: date, end: date) -> str:
    params = {
        "UniDbQuery.From": start.strftime("%d.%m.%Y"),
        "UniDbQuery.Posted": "True",
        "UniDbQuery.To": end.strftime("%d.%m.%Y"),
    }
    response = session.get(CBR_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def fetch_month(session: requests.Session, start: date, end: date) -> pd.DataFrame:
    html = get_page(session, start, end)
    result = parse_cbr_html(html)
    expected_min = 5 if (end - start).days >= 10 else 1
    if len(result) >= expected_min:
        return result[
            (result["trading_date"] >= start) & (result["trading_date"] <= end)
        ].copy()

    print(
        f"  Диапазон {start}—{end}: найдено только {len(result)} строк"
    )
    daily_rows = []
    current = start
    while current <= end:
        # ЦБ публикует кривую по рабочим дням; лишние запросы в выходные не нужны.
        if current.weekday() < 5:
            try:
                one_day = parse_cbr_html(get_page(session, current, current))
                if not one_day.empty:
                    daily_rows.extend(one_day.to_records(index=False).tolist())
            except requests.RequestException as exc:
                print(f"    Ошибка запроса {current}: {exc}")
            time.sleep(PAUSE_SECONDS)
        current += timedelta(days=1)
    if not daily_rows:
        return pd.DataFrame(columns=["trading_date", "yield_1y_pct"])
    return pd.DataFrame(
        daily_rows, columns=["trading_date", "yield_1y_pct"]
    ).drop_duplicates("trading_date", keep="last")


def read_sql_df(conn, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [item[0] for item in cur.description]
    return pd.DataFrame(rows, columns=cols)


def main():
    conn = get_connection()
    try:
        bounds = read_sql_df(conn, SQL_DATE_BOUNDS)
    finally:
        conn.close()
    if bounds.empty or pd.isna(bounds.iloc[0, 0]) or pd.isna(bounds.iloc[0, 1]):
        raise RuntimeError("Не удалось определить диапазон дат из mart.market_daily.")

    start = pd.Timestamp(bounds.iloc[0, 0]).date()
    end = pd.Timestamp(bounds.iloc[0, 1]).date()
    print("=" * 84)
    print("ЗАГРУЗКА ИСТОРИЧЕСКОЙ RF ИЗ ОФИЦИАЛЬНОЙ КРИВОЙ ЦБ РФ (СРОК 1 ГОД)")
    print(f"Период по котировкам БД: {start} → {end}")
    print(
        "Источник: cbr.ru/hd_base/zcyc_params; опубликованная доходность в % годовых."

    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; MOEXPortfolioResearch/1.0; educational research)",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        }
    )
    all_chunks = []
    chunks = list(month_chunks(start, end))
    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        try:
            frame = fetch_month(session, chunk_start, chunk_end)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Не удалось загрузить ЦБ РФ за {chunk_start}—{chunk_end}: {exc}"
            ) from exc
        if not frame.empty:
            all_chunks.append(frame)
        print(f"[{idx}/{len(chunks)}] {chunk_start:%Y-%m}: получено {len(frame)} дат")
        time.sleep(PAUSE_SECONDS)

    session.close()
    if not all_chunks:
        raise RuntimeError(
            "Не удалось извлечь ни одной строки с кривой ЦБ РФ"
        )

    curve = pd.concat(all_chunks, ignore_index=True)
    curve["trading_date"] = pd.to_datetime(
        curve["trading_date"], errors="coerce"
    ).dt.date
    curve["yield_1y_pct"] = pd.to_numeric(curve["yield_1y_pct"], errors="coerce")
    curve = curve.dropna(subset=["trading_date", "yield_1y_pct"])
    curve = curve[(curve["trading_date"] >= start) & (curve["trading_date"] <= end)]
    curve = curve.drop_duplicates("trading_date", keep="last").sort_values(
        "trading_date"
    )
    if len(curve) < 500:
        raise RuntimeError(
            f"Получено всего {len(curve)}"
        )
    if (curve["yield_1y_pct"] <= -100).any():
        raise RuntimeError("В данных обнаружена недопустимая доходность <= -100%.")
    curve["rf_daily"] = (
        np.power(
            1.0 + curve["yield_1y_pct"].to_numpy(dtype=float) / 100.0,
            1.0 / TRADING_DAYS,
        )
        - 1.0
    )
    curve["source_name"] = "CBR_ZCYC_1Y"

    csv_path = (
        Path(__file__).resolve().parents[1]
        / "research_runs/inputs/cbr_rf_zcyc_history.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(csv_path, index=False, encoding="utf-8-sig")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS mart")
            cur.execute(CREATE_SQL)
            cur.executemany(
                UPSERT_SQL,
                [
                    (
                        row.trading_date,
                        float(row.yield_1y_pct),
                        float(row.rf_daily),
                        row.source_name,
                    )
                    for row in curve.itertuples(index=False)
                ],
            )
            conn.commit()
        ff5 = read_sql_df(conn, SQL_FF5)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("\nРезультат:")
    print(f"Строк загружено/обновлено: {len(curve):,}")
    print(f"Диапазон: {curve['trading_date'].min()} → {curve['trading_date'].max()}")
    print(f"Медианная 1Y доходность: {curve['yield_1y_pct'].median():.2f}% годовых")
    print(f"Медианный daily RF: {curve['rf_daily'].median():.6f}")
    print(f"CSV: {csv_path}")
    print(
        "Таблица: mart.rf_zcyc_history"
    )

    if not ff5.empty:
        ff5["trading_date"] = pd.to_datetime(
            ff5["trading_date"], errors="coerce"
        ).dt.date
        ff5["rf"] = pd.to_numeric(ff5["rf"], errors="coerce")
        overlap = curve.merge(
            ff5.dropna(subset=["trading_date", "rf"]), on="trading_date", how="inner"
        )
        if len(overlap) >= 30:
            diff_bp = (overlap["rf_daily"] - overlap["rf"]) * 10000.0
            corr = overlap["rf_daily"].corr(overlap["rf"])
            print(
                f"Сверка с mart.ff5_factors: overlap={len(overlap)}, корреляция={corr:.4f}, MAE={diff_bp.abs().mean():.3f} б.п./день."
            )
            if corr < 0.95:
                print(
                    "CBR 1Y-derived RF заметно отличается от FF5 RF"
                )



if __name__ == "__main__":
    main()
