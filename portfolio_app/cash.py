"""Условный денежный счёт с изменением ставок, включая дни без торгов акциями."""

import numpy as np
import pandas as pd


def session_cash_returns(calendar, rates):
    dates = pd.date_range(calendar.min(), calendar.max(), freq="D")
    rates = rates.sort_index()
    aligned = (
        rates.reindex(rates.index.union(dates)).ffill().reindex(dates).astype(float)
    )
    valid = np.isfinite(aligned.to_numpy()) & (aligned.to_numpy() > -100)
    log_growth = np.log1p(np.where(valid, aligned.to_numpy() / 100, 0)) / 365.25
    prefix = pd.Series(np.r_[0, np.cumsum(log_growth[:-1])], index=dates)
    missing = pd.Series(np.r_[0, np.cumsum((~valid[:-1]).astype(int))], index=dates)
    result = np.expm1(prefix.reindex(calendar).diff())
    result[missing.reindex(calendar).diff() > 0] = np.nan
    return result


class CashCurve:
    def __init__(self, start, end, rates):
        self.dates = pd.date_range(start, end, freq="D")
        aligned = (
            rates.sort_index()
            .reindex(rates.index.union(self.dates))
            .sort_index()
            .ffill()
            .reindex(self.dates)
        )
        valid = np.isfinite(aligned) & (aligned > -100)
        logs = np.log1p(aligned.where(valid, 0).to_numpy() / 100) / 365.25
        self.prefix = pd.Series(np.r_[0, np.cumsum(logs[:-1])], index=self.dates)
        self.missing = pd.Series(
            np.r_[0, np.cumsum((~valid.to_numpy()[:-1]).astype(int))], index=self.dates
        )

    def growth(self, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if (
            end < start
            or start not in self.prefix.index
            or end not in self.prefix.index
        ):
            raise ValueError("Некорректный период начисления денежной доходности")
        if self.missing.loc[end] != self.missing.loc[start]:
            raise ValueError(
                "В периоде начисления денежной доходности отсутствует ставка"
            )
        return float(np.exp(self.prefix.loc[end] - self.prefix.loc[start]))
