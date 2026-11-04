from dataclasses import asdict, dataclass
from datetime import date
import math


# Начальные настройки продукта; будущий уровень риска ими не гарантируется.
PRESETS = {"conservative": 0.20, "balanced": 0.50, "aggressive": 0.80}


@dataclass(frozen=True)
class PortfolioConfig:
    tickers: tuple[str, ...]
    start: str
    end: str
    capital: float = 100000.0
    preset: str = "balanced"
    exposure: float | None = None
    max_weight: float = 0.10
    method: str = "equal_weight"
    rebalance_sessions: int = 21
    lookback: int = 126
    commission: float = 0.001
    slippage: float = 0.0005
    dividend_tax: float = 0.0
    return_mode: str = "total_return"

    def __post_init__(self):
        tickers = tuple(str(t).strip().upper() for t in self.tickers)
        object.__setattr__(self, "tickers", tickers)
        if (
            not tickers
            or len(set(tickers)) != len(tickers)
            or any(not t or not t.isalnum() for t in tickers)
        ):
            raise ValueError("Укажите уникальные непустые тикеры")
        if self.preset not in {*PRESETS, "custom"}:
            raise ValueError("Неизвестный профиль стратегии")
        if self.preset == "custom" and self.exposure is None:
            raise ValueError("Для пользовательского профиля нужно указать долю акций")
        e = PRESETS.get(self.preset) if self.exposure is None else self.exposure
        object.__setattr__(self, "exposure", e)
        numbers = [
            self.capital,
            e,
            self.max_weight,
            self.commission,
            self.slippage,
            self.dividend_tax,
        ]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers):
            raise ValueError("Параметры должны быть конечными числами")
        if self.capital <= 0 or not 0 <= e <= 1 or not 0 < self.max_weight <= 1:
            raise ValueError("Некорректный капитал, доля акций или лимит одной бумаги")
        if len(tickers) * self.max_weight < e - 1e-10:
            raise ValueError(
                "Недостаточно компаний для указанной доли акций и лимита бумаги"
            )
        if (
            not 0 <= self.commission + self.slippage < 0.10
            or min(self.commission, self.slippage) < 0
        ):
            raise ValueError("Некорректные торговые издержки")
        if not 0 <= self.dividend_tax < 1:
            raise ValueError("Некорректная ставка налога на дивиденды")
        if self.method not in {"equal_weight", "min_variance", "rf_only"}:
            raise ValueError("Метод не проверен для использования в приложении")
        if self.method == "rf_only" and e != 0:
            raise ValueError("Режим rf_only требует exposure=0")
        if self.return_mode not in {"total_return", "price_only"}:
            raise ValueError("Неизвестный режим доходности")
        if not isinstance(self.rebalance_sessions, int) or self.rebalance_sessions < 1:
            raise ValueError("Некорректный интервал ребалансировки")
        if not isinstance(self.lookback, int) or self.lookback < 60:
            raise ValueError("Историческое окно должно содержать не менее 60 сессий")
        if date.fromisoformat(self.start) >= date.fromisoformat(self.end):
            raise ValueError("Дата start должна предшествовать end")

    def to_dict(self):
        return asdict(self)
