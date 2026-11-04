"""Единый учёт дробных акций. Сигналы по закрытию, сделки на следующем открытии."""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio_app.math import solve_mean_variance
from .cash import CashCurve
from .dividends import DividendLedger


def risk_returns(closes, events, ledger, tax=0.0):
    ##Полные доходности для ковариации: дивиденды компенсируют механический гэп цены
    result = closes.pct_change(fill_method=None)
    previous = closes.shift(1)
    for event in events.itertuples(index=False):
        date = pd.Timestamp(event.ex_date)
        if date in result.index and event.ticker in result.columns:
            price = previous.at[date, event.ticker]
            if (
                np.isfinite(price)
                and price > 0
                and pd.notna(result.at[date, event.ticker])
            ):
                amount = (
                    event.amount
                    * (1 - tax)
                    * ledger.rate(getattr(event, "currency", "RUB"), date)
                )
                result.at[date, event.ticker] += amount / price
    return result


def calculate(config, data, progress=None, policy=None):
    """Последовательно рассчитывает портфель до конца выбранного периода."""
    simulation = PortfolioSimulation(config, data, progress)
    context = simulation.next_rebalance()
    while context is not None:
        target = None if policy is None else policy(context)
        simulation.step(target)
        context = simulation.next_rebalance()
    return simulation.results()


class PortfolioSimulation:
    """Хранит позиции и деньги между днями; расчёт выполняется обычными вызовами методов."""

    def __init__(self, config, data, progress=None):
        self.config = config
        self.progress = progress
        self.position = 0
        self.context = None
        data.validate(self.config)
        self.tickers = list(self.config.tickers)
        self.calendar = data.calendar
        self.dates = self.calendar[
            (self.calendar >= pd.Timestamp(self.config.start))
            & (self.calendar <= pd.Timestamp(self.config.end))
        ]
        if len(self.dates) < 2:
            raise ValueError("Для расчёта нужно не менее двух сессий")

        def pivot(column):
            return data.prices.pivot(
                index="trading_date", columns="ticker", values=column
            ).reindex(index=self.calendar, columns=self.tickers)

        self.closes, self.opens, self.volumes = (
            pivot("close"),
            pivot("open"),
            pivot("volume"),
        )
        self.marks = (
            self.closes.ffill()
        )  # Только оценка стоимости, пропуск не превращается в доступную цену сделки.
        rates = (
            data.rates.drop_duplicates("trading_date", keep="last")
            .set_index("trading_date")
            .yield_1y_pct.sort_index()
        )
        if not np.isfinite(rates).all() or (rates <= -100).any():
            raise ValueError("Некорректная доходность денежного счёта")
        self.cash_curve = CashCurve(self.calendar.min(), self.calendar.max(), rates)
        self.units = np.zeros(len(self.tickers))
        self.cash, self.previous_nav = (
            float(self.config.capital),
            float(self.config.capital),
        )
        self.ledger = DividendLedger(data.fx)
        self.daily, self.trades, self.allocations = [], [], []
        self.last_trade = None
        self.first_position = self.calendar.get_loc(self.dates[0])
        if self.first_position == 0:
            raise ValueError(
                "Перед началом расчёта нужна сессия для формирования решения"
            )
        self.events = data.dividends.copy()
        self.events["ex_date"] = pd.to_datetime(self.events.ex_date)
        self.events["payment_date"] = pd.to_datetime(self.events.payment_date)
        self.events["amount"] = pd.to_numeric(self.events.amount)
        if self.config.return_mode == "price_only":
            self.events = self.events.iloc[:0]
            self.history_returns = self.closes.pct_change(fill_method=None)
        else:
            historical_events = self.events[
                self.events.ticker.isin(self.tickers)
                & (self.events.ex_date <= self.dates.max())
            ]
            self.history_returns = risk_returns(
                self.closes, historical_events, self.ledger, self.config.dividend_tax
            )
            self.events = self.events[
                self.events.ticker.isin(self.tickers)
                & self.events.ex_date.between(self.dates.min(), self.dates.max())
            ]
            if not self.events.ex_date.isin(self.calendar).all():
                raise ValueError(
                    "Даты без дивиденда должны соответствовать проверенным торговым сессиям"
                )
        self.prior_date = self.calendar[self.first_position - 1]

    def next_rebalance(self):
        """Рассчитывает дни до следующей возможной ребалансировки или конца периода."""
        if self.context is not None:
            return self.context
        while self.position < len(self.dates):
            self._prepare_day()
            if self._can_rebalance():
                self.context = self._decision_context()
                return self.context
            self._finish_day()
        return None

    def step(self, proposed=None):
        """Исполняет выбранные веса и завершает текущий торговый день."""
        if self.context is None:
            raise ValueError("Сначала нужно получить дату ребалансировки")
        target = self._target_weights(proposed)
        self._execute_trades(target)
        self.context = None
        self._finish_day()

    def _prepare_day(self):
        """Начисляет деньги и дивиденды, получает текущие реальные котировки."""
        self.date = self.dates[self.position]
        position = self.calendar.get_loc(self.date)
        self.decision = self.calendar[position - 1]
        self.cash_return = self.cash_curve.growth(self.prior_date, self.date) - 1
        if not np.isfinite(self.cash_return):
            raise ValueError(
                "Отсутствует ставка, известная на начало периода начисления"
            )
        cash_before = self.cash
        principal_paid = sum(
            amount * self.ledger.rate(currency, payment)
            for payment, amount, currency in self.ledger.pending
            if payment <= self.date
        )
        self.cash = self.ledger.settle(
            self.cash, self.prior_date, self.date, self.cash_curve.growth
        )
        self.cash_income = self.cash - cash_before - principal_paid
        self.dividend_income = 0.0
        # Право определяется по позициям до покупок и продаж в дату без дивиденда.
        for event in self.events[self.events.ex_date == self.date].itertuples(
            index=False
        ):
            amount = (
                self.units[self.tickers.index(event.ticker)]
                * event.amount
                * (1 - self.config.dividend_tax)
            )
            if amount:
                currency = getattr(event, "currency", "RUB")
                self.dividend_income += amount * self.ledger.rate(currency, self.date)
                self.ledger.add(event.payment_date, amount, currency)
        self.cash = self.ledger.settle(
            self.cash, self.date, self.date, self.cash_curve.growth
        )
        self.open_quote = self.opens.loc[self.date].to_numpy(float)
        self.close_quote = self.marks.loc[self.date].to_numpy(float)
        if np.any((self.units != 0) & ~np.isfinite(self.close_quote)):
            raise ValueError("Для имеющейся позиции отсутствует оценка стоимости")
        self.tradeable = np.isfinite(self.open_quote) & (
            self.volumes.loc[self.date].fillna(0).to_numpy() > 0
        )
        self.traded = 0.0
        self.cost = 0.0

    def _can_rebalance(self):
        due = (
            self.last_trade is None
            or self.position - self.last_trade >= self.config.rebalance_sessions
        )
        if not due:
            return False
        self.available = (
            self.marks.loc[self.decision].notna().to_numpy() & self.tradeable
        )
        if self.config.method == "min_variance":
            history = self.history_returns.loc[: self.decision].tail(
                self.config.lookback
            )
            self.available &= history.notna().sum().to_numpy() >= 60
        if np.any((self.units != 0) & ~self.tradeable) or not self.tradeable.any():
            return False
        if self.available.sum() * self.config.max_weight < self.config.exposure - 1e-10:
            raise ValueError(
                f"Недостаточно торгуемых компаний на дату {self.date.date()} для заданной доли акций"
            )
        return True

    def _decision_context(self):
        """Возвращает стратегии только доступную к решению информацию."""
        open_marks = np.where(self.tradeable, self.open_quote, self.close_quote)
        self.stock_values = np.where(
            self.units != 0, self.units * np.nan_to_num(open_marks), 0
        )
        self.nav = float(
            self.cash + self.stock_values.sum() + self.ledger.value(self.date)
        )
        if self.nav <= 0:
            raise ValueError("Средства портфеля исчерпаны")
        return {
            "decision_date": self.decision,
            "execution_date": self.date,
            "tickers": tuple(self.tickers),
            "available": self.available.copy(),
            "closes": self.closes.loc[: self.decision],
            "returns": self.history_returns.loc[: self.decision],
            "previous_nav": self.previous_nav,
            "stock_weights": self.units
            * np.nan_to_num(self.marks.loc[self.decision].to_numpy(float))
            / self.previous_nav,
        }

    def _target_weights(self, proposed):
        indices = np.flatnonzero(self.available)
        target = np.zeros(len(self.tickers))
        if proposed is not None:
            target = np.asarray(proposed, dtype=float)
            if (
                target.shape != (len(self.tickers),)
                or not np.isfinite(target).all()
                or (target < -1e-10).any()
                or (target > self.config.max_weight + 1e-8).any()
                or np.any(np.abs(target[~self.available]) > 1e-10)
                or not np.isclose(target.sum(), self.config.exposure, atol=1e-8)
            ):
                raise ValueError(
                    "Исследовательская стратегия вернула недопустимые веса акций"
                )
            return np.maximum(target, 0)
        if self.config.exposure == 0:
            return target
        if self.config.method != "min_variance":
            target[indices] = self.config.exposure / len(indices)
            return target

        history = self.history_returns.loc[: self.decision].tail(self.config.lookback)
        complete = history.iloc[:, indices].dropna()
        if len(complete) < 60:
            raise ValueError("Недостаточно общей истории для оценки ковариации")
        covariance = LedoitWolf().fit(complete.to_numpy()).covariance_ * 252
        weights, _, _ = solve_mean_variance(
            np.zeros(len(indices)),
            covariance,
            0,
            1,
            self.config.max_weight,
            1 - self.config.exposure,
            1 - self.config.exposure,
        )
        target[indices] = weights
        return target

    def _execute_trades(self, target):
        # Ожидаемые дивиденды входят в активы, но до выплаты не могут финансировать сделки.
        liquid_nav = float(self.cash + self.stock_values.sum())
        # Перед расчётом количества акций резервируем издержки в стоимости портфеля.
        fee_rate = self.config.commission + self.config.slippage
        post_cost_nav = liquid_nav
        for _ in range(100):
            desired = target * post_cost_nav
            estimate = fee_rate * np.abs(desired - self.stock_values).sum()
            updated = liquid_nav - estimate
            if abs(updated - post_cost_nav) < 1e-9 * max(self.nav, 1):
                post_cost_nav = updated
                break
            post_cost_nav = updated
        desired = target * post_cost_nav
        change = desired - self.stock_values
        self.traded = float(np.abs(change).sum())
        self.cost = self.traded * fee_rate
        self.cash -= float(change.sum()) + self.cost
        if self.cash < -1e-6:
            raise ValueError("Недостаточно доступных денежных средств")
        self.cash = max(0.0, self.cash)
        for i in range(len(self.tickers)):
            if abs(change[i]) > 1e-8:
                self.trades.append(
                    {
                        "date": self.date,
                        "decision_date": self.decision,
                        "ticker": self.tickers[i],
                        "notional": float(change[i]),
                        "reference_open": float(self.open_quote[i]),
                        "cost": float(abs(change[i]) * fee_rate),
                    }
                )
            self.units[i] = desired[i] / self.open_quote[i] if target[i] > 0 else 0.0
        self.last_trade = self.position

    def _finish_day(self):
        """Записывает дневной результат и переходит к следующему дню."""
        self.stock_values = self.units * np.nan_to_num(self.close_quote)
        pending = self.ledger.value(self.date)
        self.nav = float(self.cash + self.stock_values.sum() + pending)
        if self.nav <= 0 or not np.isfinite(self.nav):
            raise ValueError("Некорректная стоимость портфеля")
        self.daily.append(
            {
                "date": self.date,
                "nav": self.nav,
                "net_return": self.nav / self.previous_nav - 1,
                "cash": self.cash,
                "dividend_receivable": pending,
                "dividend_income": self.dividend_income,
                "cash_income": self.cash_income,
                "rf_return": self.cash_return,
                "cost": self.cost,
                "traded_notional": self.traded,
                "nontradable_positions": int(
                    np.sum((self.units != 0) & ~self.tradeable)
                ),
                "stale_valuations": int(
                    np.sum(
                        (self.units != 0) & self.closes.loc[self.date].isna().to_numpy()
                    )
                ),
            }
        )
        for ticker, quantity, value in zip(self.tickers, self.units, self.stock_values):
            self.allocations.append(
                {
                    "date": self.date,
                    "ticker": ticker,
                    "units": float(quantity),
                    "weight": float(value / self.nav),
                }
            )
        self.previous_nav, self.prior_date = self.nav, self.date
        if self.progress and (
            self.position % 21 == 0 or self.position == len(self.dates) - 1
        ):
            self.progress(
                (self.position + 1) / len(self.dates),
                f"Рассчитана дата {self.date.date()}",
            )
        self.position += 1

    def results(self):
        """Собирает дневную историю, операции, веса и итоговые метрики."""
        daily = pd.DataFrame(self.daily)
        curve = np.r_[self.config.capital, daily.nav.to_numpy()]
        drawdown = curve / np.maximum.accumulate(curve) - 1
        years = (self.dates[-1] - self.calendar[self.first_position - 1]).days / 365.25
        sessions_per_year = len(self.dates) / years
        returns = daily.net_return.to_numpy()
        excess = returns - daily.rf_return.to_numpy()
        excess_std = float(np.std(excess, ddof=1))
        metrics = {
            "total_return": float(curve[-1] / self.config.capital - 1),
            "cagr": float((curve[-1] / self.config.capital) ** (1 / years) - 1),
            "max_drawdown": float(drawdown.min()),
            "cost": float(daily.cost.sum()),
            "volatility": float(np.std(returns, ddof=1) * np.sqrt(sessions_per_year)),
            "sharpe_excess_cash": float(
                np.mean(excess) / excess_std * np.sqrt(sessions_per_year)
            )
            if excess_std > 1e-12
            else None,
            "sessions": len(self.dates),
            "return_mode": self.config.return_mode,
        }
        return {
            "daily": daily,
            "trades": pd.DataFrame(self.trades),
            "weights": pd.DataFrame(self.allocations),
            "metrics": pd.DataFrame([metrics]),
        }
