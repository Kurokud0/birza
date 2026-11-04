"""Согласование источников и явный импорт дивидендов без изменения исходных таблиц."""

from datetime import datetime, timezone
import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import load_snapshot, save_snapshot, validate_dividends


def download_tbank_dividends(tickers, start, end, output_root="research_runs"):
    """Загружает события для проверки. Нужен локально настроенный токен только для чтения.

    Загрузка не подтверждает покрытие и не восстанавливает даты права/выплаты по дате реестра."""
    tickers = [ticker.strip().upper() for ticker in tickers]
    if any(
        not re.fullmatch(r"[A-Z0-9]+", ticker) for ticker in tickers
    ) or date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("Некорректные тикеры или диапазон дат")
    token = os.environ.get("TINVEST_TOKEN")
    if not token:
        raise ValueError(
            "!!!Задайте TINVEST_TOKEN локально, не помещайте токены API в настройки проекта или чат!!!"
        )
    from uuid import uuid4

    root = Path(output_root) / str(uuid4())
    root.mkdir(parents=True)
    base = "https://invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/"
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {token}"})

        def call(method, body):
            response = session.post(base + method, json=body, timeout=30)
            if not response.ok:
                raise RuntimeError(
                    f"Т-Банк, метод {method}: код HTTP {response.status_code}"
                )
            return response.json()

        for ticker in tickers:
            instrument = call(
                "GetInstrumentBy",
                {
                    "idType": "INSTRUMENT_ID_TYPE_TICKER",
                    "classCode": "TQBR",
                    "id": "T" if ticker == "TCSG" else ticker,
                },
            )["instrument"]
            payload = call(
                "GetDividends",
                {
                    "figi": instrument["figi"],
                    "from": start + "T00:00:00Z",
                    "to": end + "T23:59:59Z",
                },
            )
            if "dividends" not in payload:
                raise ValueError(
                    f"Отсутствует блок дивидендов для {ticker}; ответ не подтверждает отсутствие выплат"
                )
            result = {
                "ticker": ticker,
                "instrument": instrument,
                "source": base + "GetDividends",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "requested_start": start,
                "requested_end": end,
                "review_status": "unverified",
                "response": payload,
            }
            (root / f"{ticker}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return root.resolve()


def _candles(ticker, start, end):
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"
    response = requests.get(
        url, params={"from": start, "till": end, "interval": 24}, timeout=30
    )
    response.raise_for_status()
    payload = response.json()["candles"]
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    frame["trading_date"] = pd.to_datetime(frame["begin"]).dt.normalize()
    return frame, response.url


def reconcile_tcsg(data):
    """Сверяет пересекающиеся официальные цены и приводит старый участок к единицам T."""
    old, old_url = _candles("TCSG", "2024-10-01", "2024-11-27")
    new, new_url = _candles("T", "2024-10-01", "2024-11-27")
    overlap = old.merge(new, on="trading_date", suffixes=("_old", "_new"))
    if len(overlap) < 10:
        raise ValueError(
            "Недостаточно пересекающихся официальных наблюдений для согласования тикеров"
        )
    ratios = np.concatenate(
        [overlap[f"{col}_new"] / overlap[f"{col}_old"] for col in ["open", "close"]]
    )
    factor = float(np.median(ratios))
    if (
        not np.isfinite(factor)
        or factor <= 0
        or not np.allclose(ratios, factor, rtol=1e-8)
    ):
        raise ValueError(
            "Масштабы цен тикеров не связаны одним подтверждённым коэффициентом"
        )
    original = data.prices[
        (data.prices.ticker == "TCSG") & (data.prices.trading_date < "2024-11-28")
    ]
    check = original.merge(
        old[["trading_date", "open", "close"]],
        on="trading_date",
        suffixes=("_db", "_source"),
    )
    if len(check) < 10 or not all(
        np.allclose(check[f"{col}_db"], check[f"{col}_source"], rtol=1e-8)
        for col in ["open", "close"]
    ):
        raise ValueError(
            "Сохранённые старые цены не совпадают с проверенным источником; повторную корректировку выполнять нельзя"
        )
    data.prices = data.prices.copy()
    mask = (data.prices.ticker == "TCSG") & (data.prices.trading_date < "2024-11-28")
    data.prices.loc[mask, ["open", "close"]] *= factor
    data.metadata = {
        **data.metadata,
        "price_basis": "Исходные цены; старый участок TCSG согласован с единицами цены T",
    }
    data.metadata["adjustments"] = [
        *data.metadata.get("adjustments", []),
        {
            "ticker": "TCSG",
            "before": "2024-11-28",
            "price_factor": factor,
            "verified_overlap_sessions": len(overlap),
            "old_source": old_url,
            "new_source": new_url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "volume": "Без изменений: объёмы официальных свечей сохранены в исходном масштабе источника",
            "evidence": json.loads(
                overlap[["trading_date", "close_old", "close_new"]].to_json(
                    orient="records", date_format="iso"
                )
            ),
        },
    ]
    return data


def import_dividends(snapshot_path, csv_path, coverage_path, output_root):
    """Импортирует проверенные события в единицах акций, соответствующих ценам снимка.

    Пропуски дат и выплат не восстанавливаются из пустого ответа.
    Проверка явно подтверждает полноту, включая компании без дивидендов."""
    data = load_snapshot(snapshot_path)
    events = pd.read_csv(csv_path, parse_dates=["ex_date", "payment_date"])
    validate_dividends(events)
    review = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    if review.get("price_basis") != data.metadata["price_basis"]:
        raise ValueError(
            "Единицы дивиденда на акцию должны соответствовать масштабу цен снимка"
        )
    if not review.get("reviewed_by") or not review.get("source"):
        raise ValueError("Нужно указать автора проверки и источник")
    for ticker, item in review["coverage"].items():
        if not item.get("verified") or pd.Timestamp(item["start"]) > pd.Timestamp(
            item["end"]
        ):
            raise ValueError(f"Некорректное покрытие для {ticker}")
    data.dividends = events
    data.metadata = {
        **data.metadata,
        "dividend_coverage": review["coverage"],
        "dividend_review": review,
    }
    return save_snapshot(data, output_root)
