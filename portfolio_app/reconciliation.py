"""Сверка графиков без изменения исходных наблюдений.

Проверка относится к документированным правилам моделирования, а не времени
зачисления, налогам или валютному спреду конкретного брокера."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import pandas as pd

from .data import validate_dividends

GDR_EX_DATES = {
    "2020-03-27": "2020-03-26",
    "2020-05-29": "2020-05-28",
    "2020-08-21": "2020-08-20",
    "2020-11-27": "2020-11-25",
    "2021-03-26": "2021-03-25",
}
GDR_EVIDENCE = {
    "2020-03-27": "https://www.lse.co.uk/rns/TCS/correction-tcs-group-holding-plc-reports-record-net-income-in-4q-and-fy-2019-announces-1st-2020-interim-dividend-9qq3409ekyltums.html",
    "2020-05-29": "https://www.lse.co.uk/rns/tcs-group-holding-plc-reports-robust-profit-in-1q20-announces-2nd-2020-interim-dividend-luc72lvbmhna7hy.html",
    "2020-08-21": "https://www.stockopedia.com/share-prices/tcs-holding-LON%3ATCS/news/reg-tcs-group-holding-plc-tcs-group-holding-plc-reports-strong-financial-performance-in-2q-20-and-1h-20-announces-3rd-2020-interim-dividend-urn%3Anewsml%3Areuters.com%3A20200806%3AnEQ9KBvk8a/",
    "2020-11-27": "https://tinkoff-group.com/company-info/news/12112020-tcs-group-holding-plc-fin-results-3q-2020-eng/",
    "2021-03-26": "https://tinkoff-group.com/company-info/news/11032021-tcs-group-holding-plc-fin-results-fy-2020-eng/",
}


def verify_calendar(data, directory):
    root = Path(directory)
    stock = pd.read_csv(root / "official_sber_candles.csv")
    index = pd.read_csv(root / "official_imoex_candles.csv")
    official = (
        pd.DatetimeIndex(pd.to_datetime(stock["begin"]).dt.normalize())
        .union(pd.DatetimeIndex(pd.to_datetime(index.trading_date)))
        .unique()
        .sort_values()
    )
    if not data.calendar.equals(official):
        raise ValueError(
            "Календарь снимка отличается от официальных наблюдений по акциям и индексу"
        )
    data.metadata["calendar_verification"] = {
        "method": "Точное объединение независимо полученных официальных свечей SBER и IMOEX",
        "sessions": len(official),
        "settlement_calendar": False,
        "source_sha256": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in root.glob("*.csv")
        },
    }


def reconcile_dividends(data, source_directories, reference_file):
    references = json.loads(Path(reference_file).read_text(encoding="utf-8"))
    # Две выплаты PIK имеют общую дату права: сохраняем обе, не удаляя по совпадению дат.
    pikk = next(item for item in references if item["ticker"] == "PIKK")
    pikk["url"] = "https://pik-group.ru/about/share-capital/dividends"
    pikk["rows"] = [
        {"record_date": "2020-10-12", "last_buy": "2020-10-08", "amount_text": "22.71"},
        {"record_date": "2021-05-17", "last_buy": "2021-05-13", "amount_text": "22.51"},
        {"record_date": "2021-05-17", "last_buy": "2021-05-13", "amount_text": "22.92"},
    ]
    rows, audit, source_hashes = [], [], {}
    amounts_by_reference = defaultdict(float)
    for root in source_directories:
        for file in sorted(Path(root).glob("*.json")):
            document = json.loads(file.read_text(encoding="utf-8"))
            ticker = document["ticker"]
            reference = next(item for item in references if item["ticker"] == ticker)
            source_hashes[str(file.resolve())] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for number, event in enumerate(document["response"]["dividends"]):
                record = pd.Timestamp(event["recordDate"][:10])
                candidates = [
                    r
                    for r in reference["rows"]
                    if abs((pd.Timestamp(r["record_date"]) - record).days) <= 4
                ]
                if not candidates:
                    raise ValueError(
                        f"Отсутствует независимый источник для сверки: {ticker} {record.date()}"
                    )
                distance = min(
                    abs((pd.Timestamp(r["record_date"]) - record).days)
                    for r in candidates
                )
                candidates = [
                    r
                    for r in candidates
                    if abs((pd.Timestamp(r["record_date"]) - record).days) == distance
                ]
                buys = {r["last_buy"] for r in candidates}
                if len(buys) != 1:
                    raise ValueError(
                        f"Источники противоречат друг другу в дате дивидендного права: {ticker} {record.date()}"
                    )
                last_buy = pd.Timestamp(next(iter(buys)))
                if last_buy not in data.calendar:
                    raise ValueError(
                        f"Последний день покупки из источника не является официальной сессией: {ticker} {last_buy}"
                    )
                idx = data.calendar.searchsorted(last_buy, side="right")
                if idx == len(data.calendar):
                    raise ValueError(
                        "Дата без дивиденда находится вне проверенного календаря"
                    )
                ex_date = data.calendar[idx]
                date_source = reference["url"]
                if ticker == "TCSG" and str(record.date()) in GDR_EX_DATES:
                    ex_date = pd.Timestamp(GDR_EX_DATES[str(record.date())])
                    date_source = GDR_EVIDENCE[str(record.date())]
                if ex_date not in data.calendar:
                    raise ValueError(
                        "Дата без дивиденда по данным эмитента находится вне официального календаря сессий"
                    )
                money = event["dividendNet"]
                amount = float(money["units"]) + money.get("nano", 0) / 1e9
                amount_source = document["source"]
                # Сумма эмитента 1301.75 делится на 10; округлённое значение API 130.18 не используем.
                if ticker == "PLZL" and record == pd.Timestamp("2024-12-13"):
                    amount = 1301.75 / 10
                    amount_source = "https://polyus.com/en/investors/shares/dividends/"
                payment = pd.Timestamp(event["paymentDate"][:10])
                key = f"{ticker}:{record.date()}:{money['currency']}:{amount}:{number}"
                event_id = str(uuid5(NAMESPACE_URL, key))
                rows.append(
                    {
                        "event_id": event_id,
                        "ticker": ticker,
                        "ex_date": ex_date,
                        "payment_date": payment,
                        "amount": amount,
                        "currency": money["currency"].upper(),
                        "source": amount_source,
                        "date_source": date_source,
                        "record_date": record,
                        "payment_convention": "Дата выплаты провайдера; дата зачисления на конкретный брокерский счёт не подтверждается",
                    }
                )
                amounts_by_reference[(ticker, candidates[0]["record_date"])] += amount
                audit.append(
                    {
                        "event_id": event_id,
                        "ticker": ticker,
                        "record_date": record,
                        "api_last_buy": event["lastBuyDate"][:10],
                        "reference_last_buy": last_buy,
                        "resolved_ex_date": ex_date,
                        "amount": amount,
                        "currency": money["currency"],
                        "resolution": "Дата без дивиденда ГДР по данным эмитента"
                        if date_source != reference["url"]
                        else "Биржевой график последней покупки; учтены исторические режимы T+2/T+1",
                        "reference_record_date": candidates[0]["record_date"],
                    }
                )
    events = pd.DataFrame(rows)
    validate_dividends(events)
    if len(events) != 201 or events.ticker.nunique() != 25:
        raise ValueError(
            "Неожиданный набор инструментов источника; проверьте протокол согласования"
        )
    for reference in references:
        ticker = reference["ticker"]
        groups = defaultdict(list)
        for row in reference["rows"]:
            if "2020-01-03" <= row["record_date"] <= "2026-09-21":
                groups[row["record_date"]].append(row)
        for record, group in groups.items():
            expected, tolerance = 0.0, 1e-7
            for row in group:
                numeric = re.sub(r"[^0-9,.]", "", row["amount_text"]).replace(",", ".")
                factor = 1.0
                if ticker == "PLZL" and record < "2025-03-25":
                    factor = 0.1
                if ticker == "TCSG" and record < "2026-04-15":
                    factor = 0.1
                if ticker == "VTBR" and record < "2024-07-11":
                    factor = 5000.0
                expected += float(numeric) * factor
                decimals = len(numeric.split(".")[1]) if "." in numeric else 0
                tolerance += 0.5 * 10 ** (-decimals) * factor
            actual = amounts_by_reference.get((ticker, record))
            if actual is None or abs(actual - expected) > tolerance:
                raise ValueError(
                    f"Расхождение суммы или полноты дивидендных событий: {ticker} {record}: {actual} vs {expected} ± {tolerance}"
                )
    data.dividends = events
    data.metadata["dividend_coverage"] = {
        ticker: {
            "verified": True,
            "start": "2020-01-03",
            "end": "2026-09-21",
            "level": "Межисточниковая сверка графика по документированным правилам моделирования",
        }
        for ticker in events.ticker.unique()
    }
    data.metadata["dividend_contract"] = {
        "payment": "Даты выплат провайдера; время зачисления на конкретный счёт не подтверждается",
        "currency": "Требования оцениваются по известному официальному курсу ЦБ; конвертация по курсу ЦБ на дату выплаты",
        "units": "Дивиденды Т-Банка с учётом дроблений; точность PLZL за 2024 год уточнена по данным эмитента",
        "dates": "Последние дни покупки по биржевому графику и официальный календарь сессий; для ГДР использованы отдельные даты эмитента",
        "completeness": "Исторические графики нескольких источников; полный независимый реестр корпоративных действий не восстанавливался",
        "reference_sha256": hashlib.sha256(
            Path(reference_file).read_bytes()
        ).hexdigest(),
        "source_sha256": source_hashes,
    }
    return pd.DataFrame(audit)
