"""Общие портфельные вычисления. Здесь используются абсолютные веса."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def maximum_drawdown(returns):
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return np.nan
    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("Некорректные простые доходности")
    nav = np.r_[1.0, np.cumprod(1 + values)]
    return float(np.min(nav / np.maximum.accumulate(nav) - 1))


def solve_mean_variance(
    expected_returns,
    covariance,
    rf,
    risk_aversion,
    max_weight,
    max_rf,
    min_rf=0.0,
    initial=None,
):
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    n = len(mu)
    if (
        n == 0
        or sigma.shape != (n, n)
        or not np.isfinite(mu).all()
        or not np.isfinite(sigma).all()
    ):
        raise ValueError(
            "Ожидаются конечные доходности и ковариационная матрица соответствующего размера"
        )
    if not 0 <= min_rf <= max_rf <= 1 or not 0 < max_weight <= 1 or risk_aversion <= 0:
        raise ValueError("Некорректные ограничения риска или весов")
    low, high = 1 - max_rf, 1 - min_rf
    if n * max_weight < low - 1e-10:
        raise ValueError(
            "Недостаточно акций для заданной экспозиции и ограничения веса"
        )
    sigma = (sigma + sigma.T) / 2
    if np.linalg.eigvalsh(sigma).min() < -1e-8:
        raise ValueError(
            "Ковариационная матрица должна быть положительно полуопределённой"
        )
    if initial is None:
        initial = np.full(n, low / n)
    else:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (n,) or not np.isfinite(initial).all():
            raise ValueError("Некорректные начальные веса")
        if (
            initial.min() < 0
            or initial.max() > max_weight
            or not low <= initial.sum() <= high
        ):
            initial = np.full(n, low / n)
    excess = mu - rf
    ones = np.ones(n)

    def objective(w):
        return float(-w @ excess + risk_aversion / 2 * (w @ sigma @ w))

    def gradient(w):
        return risk_aversion * sigma @ w - excess

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0, max_weight)] * n,
        constraints=[
            {"type": "ineq", "fun": lambda w: high - w.sum(), "jac": lambda w: -ones},
            {"type": "ineq", "fun": lambda w: w.sum() - low, "jac": lambda w: ones},
        ],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Не удалось оптимизировать портфель: {result.message}")
    weights = np.clip(result.x, 0, max_weight)
    if not low - 1e-6 <= weights.sum() <= high + 1e-6:
        raise RuntimeError("Оптимизатор вернул недопустимую долю акций")
    return weights, max(0.0, 1 - float(weights.sum())), result


def holding_period(stock_returns, rf_returns, weights, rf_weight):
    """Однократная покупка с последующим дрейфом долей без скрытой ежедневной ребалансировки."""
    returns = np.asarray(stock_returns, dtype=float)
    cash_returns = np.asarray(rf_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        returns.ndim != 2
        or returns.shape != (len(cash_returns), len(weights))
        or not len(returns)
    ):
        raise ValueError("Несогласованные размеры данных периода удержания")
    if not np.isfinite(returns).all() or not np.isfinite(cash_returns).all():
        raise ValueError("Доходности должны быть конечными числами")
    if (returns < -1).any() or (cash_returns < -1).any():
        raise ValueError("Простая доходность не может быть ниже -100%")
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(rf_weight)
        or (weights < 0).any()
        or rf_weight < 0
    ):
        raise ValueError("Ожидаются конечные неотрицательные веса без коротких позиций")
    if not np.isclose(weights.sum() + rf_weight, 1.0):
        raise ValueError("Сумма весов акций и денег должна равняться единице")
    stock_values = np.cumprod(1 + returns, axis=0) * weights
    cash_values = np.cumprod(1 + cash_returns) * rf_weight
    nav = stock_values.sum(axis=1) + cash_values
    prior_nav = np.r_[1.0, nav[:-1]]
    if (nav <= 0).any():
        raise ValueError("Стоимость портфеля достигла нуля")
    prior_stocks = np.vstack([weights, stock_values[:-1]])
    prior_cash = np.r_[rf_weight, cash_values[:-1]]
    return {
        "gross_return": nav / prior_nav - 1,
        "stock_return": (prior_stocks * returns).sum(axis=1) / prior_nav,
        "rf_weight": prior_cash / prior_nav,
        "final_weights": stock_values[-1] / nav[-1],
        "final_rf": float(cash_values[-1] / nav[-1]),
    }


def validate_price_jumps(frame, threshold=0.50):
    prices = frame[["ticker", "trading_date", "close"]].copy()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.sort_values(["ticker", "trading_date"])
    changes = prices.groupby("ticker", sort=False)["close"].pct_change(fill_method=None)
    suspect = prices.loc[changes.abs() > threshold].copy()
    if not suspect.empty:
        suspect["return"] = changes.loc[suspect.index]
        examples = suspect.head(5).to_dict("records")
        raise ValueError(
            f"Необъяснённые изменения цены выше {threshold:.0%}: {examples}. "
            "Проверьте исходные цены, смену тикеров и корпоративные действия."
        )
