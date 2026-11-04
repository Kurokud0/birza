"""Версионные снимки входов. Исходные рыночные таблицы используются только на чтение."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from portfolio_app.math import validate_price_jumps
from portfolio_app.db import get_connection

DIVIDEND_COLUMNS = ["event_id", "ticker", "ex_date", "payment_date", "amount", "source"]


@dataclass
class MarketData:
    prices: pd.DataFrame
    rates: pd.DataFrame
    calendar: pd.DatetimeIndex
    dividends: pd.DataFrame
    metadata: dict
    fx: pd.DataFrame | None = None

    def validate(self, config):
        frame = self.prices
        if (
            self.calendar.empty
            or pd.Timestamp(config.start) < self.calendar.min()
            or pd.Timestamp(config.end) > self.calendar.max()
        ):
            raise ValueError(
                "Запрошенный период находится за пределами проверенного календаря - обновите данные"
            )
        if frame.duplicated(["ticker", "trading_date"]).any():
            raise ValueError("Обнаружены повторяющиеся рыночные записи")
        if not self.calendar.is_unique or not self.calendar.is_monotonic_increasing:
            raise ValueError(
                "Календарь должен быть отсортирован и не содержать повторов"
            )
        if not frame.trading_date.isin(self.calendar).all():
            raise ValueError("Котировки выходят за пределы календаря сессий")
        if not set(config.tickers).issubset(set(frame.ticker)):
            raise ValueError("Неизвестные компании или отсутствующая история")
        selected = frame[
            frame.ticker.isin(config.tickers)
            & (frame.trading_date <= pd.Timestamp(config.end))
        ]
        validate_price_jumps(selected)
        for col in ["open", "close"]:
            if not np.isfinite(selected[col]).all() or (selected[col] <= 0).any():
                raise ValueError("Цены должны быть конечными положительными числами")
        if (selected.volume < 0).any():
            raise ValueError("Отрицательный объём торгов")
        if config.return_mode == "total_return" and config.exposure > 0:
            coverage = self.metadata.get("dividend_coverage", {})
            for ticker in config.tickers:
                item = coverage.get(ticker, {})
                if (
                    not item.get("verified")
                    or item.get("start", "9999") > config.start
                    or item.get("end", "0000") < config.end
                ):
                    raise ValueError(
                        f"Требуется подтверждённое дивидендное покрытие для {ticker}; ценовой режим нужно выбрать явно"
                    )
        validate_dividends(self.dividends)


def validate_dividends(frame):
    if not set(DIVIDEND_COLUMNS).issubset(frame.columns):
        raise ValueError("Отсутствуют обязательные поля дивидендов")
    if frame.empty:
        return
    if (
        frame.event_id.duplicated().any()
        or frame[["event_id", "ticker", "source"]].isna().any().any()
    ):
        raise ValueError("Повторяющиеся или неидентифицированные дивидендные события")
    if (
        frame[["event_id", "ticker", "source"]]
        .astype(str)
        .apply(lambda s: s.str.strip().eq(""))
        .any()
        .any()
    ):
        raise ValueError("Идентификатор и источник дивиденда не могут быть пустыми")
    amount = pd.to_numeric(frame.amount, errors="coerce")
    if not np.isfinite(amount).all() or (amount <= 0).any():
        raise ValueError(
            "Дивиденд на акцию должен быть положительным и соответствовать масштабу цены"
        )
    if (
        "currency" in frame
        and not frame.currency.astype(str).str.fullmatch(r"[A-Za-z]{3}").all()
    ):
        raise ValueError("Нужно явно указать код валюты дивиденда по ISO")
    ex = pd.to_datetime(frame.ex_date, errors="raise")
    pay = pd.to_datetime(frame.payment_date, errors="raise")
    if ex.isna().any() or pay.isna().any() or (pay < ex).any():
        raise ValueError(
            "Даты дивиденда должны быть известны; выплата не может предшествовать дате без дивиденда"
        )


def read_market():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")

            def query(sql):
                cur.execute(sql)
                return pd.DataFrame(
                    cur.fetchall(), columns=[c.name for c in cur.description]
                )

            prices = query(
                "SELECT ticker, trading_date, open, close, volume FROM mart.market_daily ORDER BY ticker,trading_date"
            )
            rates = query(
                "SELECT trading_date,yield_1y_pct FROM mart.rf_zcyc_history ORDER BY trading_date"
            )
            sessions = query(
                "SELECT DISTINCT trading_date FROM core.market_indices WHERE index_code='IMOEX' ORDER BY trading_date"
            )
            fx = query(
                "SELECT currency,date AS trading_date,value AS rate FROM core.currency_rates WHERE currency IN ('USD','EUR') ORDER BY currency,date"
            )
    finally:
        conn.rollback()
        conn.close()
    prices.trading_date = pd.to_datetime(prices.trading_date)
    rates.trading_date = pd.to_datetime(rates.trading_date)
    for col in ["open", "close", "volume"]:
        prices[col] = pd.to_numeric(prices[col])
    rates.yield_1y_pct = pd.to_numeric(rates.yield_1y_pct)
    fx.trading_date = pd.to_datetime(fx.trading_date)
    fx.rate = pd.to_numeric(fx.rate)
    calendar = (
        pd.DatetimeIndex(prices.trading_date)
        .union(pd.DatetimeIndex(sessions.trading_date))
        .unique()
        .sort_values()
    )
    return MarketData(
        prices,
        rates,
        calendar,
        pd.DataFrame(columns=DIVIDEND_COLUMNS),
        {
            "price_basis": "source_prices",
            "calendar": "Объединение наблюдаемых сессий акций и IMOEX; календарь не дополнен вымышленными будними днями",
            "dividend_coverage": {},
            "adjustments": [],
        },
        fx,
    )


def save_snapshot(data, directory="data_snapshots"):
    path = Path(directory) / str(uuid4())
    path.mkdir(parents=True)
    frames = {
        "prices": data.prices,
        "rates": data.rates,
        "calendar": pd.DataFrame({"trading_date": data.calendar}),
        "dividends": data.dividends,
    }
    if data.fx is not None:
        frames["fx"] = data.fx
    hashes = {}
    for name, frame in frames.items():
        file = path / f"{name}.csv"
        frame.to_csv(file, index=False)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    metadata = {**data.metadata, "snapshot_id": path.name, "sha256": hashes}
    (path / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path.resolve()


def load_snapshot(directory):
    path = Path(directory)
    meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name in ["prices.csv", "rates.csv", "calendar.csv", "dividends.csv"]:
        if (
            hashlib.sha256((path / name).read_bytes()).hexdigest()
            != meta["sha256"][name]
        ):
            raise ValueError(f"Входной снимок: несовпадение хеша: {name}")
    prices = pd.read_csv(path / "prices.csv", parse_dates=["trading_date"])
    rates = pd.read_csv(path / "rates.csv", parse_dates=["trading_date"])
    calendar = pd.DatetimeIndex(
        pd.read_csv(path / "calendar.csv", parse_dates=["trading_date"]).trading_date
    )
    divs = pd.read_csv(path / "dividends.csv", parse_dates=["ex_date", "payment_date"])
    fx = None
    if "fx.csv" in meta["sha256"]:
        if (
            hashlib.sha256((path / "fx.csv").read_bytes()).hexdigest()
            != meta["sha256"]["fx.csv"]
        ):
            raise ValueError("Входной снимок: несовпадение хеша: fx.csv")
        fx = pd.read_csv(path / "fx.csv", parse_dates=["trading_date"])
    return MarketData(prices, rates, calendar, divs, meta, fx)
