"""Дивидендные требования и конвертация по известным на соответствующую дату официальным курсам."""

import numpy as np
import pandas as pd


class DividendLedger:
    def __init__(self, fx=None):
        self.pending = []
        self.fx = {}
        if fx is not None:
            for currency, frame in fx.groupby("currency"):
                frame = frame.sort_values("trading_date")
                if (
                    frame.trading_date.duplicated().any()
                    or not np.isfinite(frame.rate).all()
                    or (frame.rate <= 0).any()
                ):
                    raise ValueError(
                        "Некорректные или повторяющиеся валютные наблюдения"
                    )
                self.fx[currency.upper()] = frame.set_index("trading_date").rate

    def rate(self, currency, date):
        if currency.upper() == "RUB":
            return 1.0
        series = self.fx.get(currency.upper())
        if series is None:
            raise ValueError(f"Отсутствует история валютных курсов: {currency}")
        position = series.index.searchsorted(pd.Timestamp(date), side="right") - 1
        if position < 0:
            raise ValueError(f"Нет известного курса на дату {date} для {currency}")
        return float(series.iloc[position])

    def add(self, payment_date, amount, currency="RUB"):
        self.pending.append(
            (pd.Timestamp(payment_date), float(amount), currency.upper())
        )

    def value(self, date):
        return sum(
            amount * self.rate(currency, date) for _, amount, currency in self.pending
        )

    def settle(self, cash, start, end, growth):
        ##Начисляет денежный доход между реальными датами выплат, включая неторговые дни
        remaining = []
        cursor = pd.Timestamp(start)
        for payment, amount, currency in sorted(self.pending):
            if payment <= end:
                effective = max(cursor, payment)
                cash *= growth(cursor, effective)
                cash += amount * self.rate(currency, payment)
                cursor = effective
            else:
                remaining.append((payment, amount, currency))
        cash *= growth(cursor, end)
        self.pending = remaining
        return cash
