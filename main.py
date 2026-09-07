from moex import load_data_moex, save_to_raw
from cbr import (
    load_data_cbr,
    load_data_metals,
    load_data_key_rate,
    save_currency_to_raw,
    save_metals_to_raw,
    save_key_rate_to_raw
)


TICKERS = [
    "SBER",
    "GAZP",
    "LKOH",
    "GMKN",
    "YDEX",
    "ROSN",
    "NVTK",
    "TATN",
    "MGNT",
    "MTSS"
]

CURRENCIES = [
    "USD",
    "EUR",
    "CNY",
    "GBP",
    "JPY",
    "CHF",
    "TRY",
    "AED",
    "KZT",
    "INR"
]

DATE_FROM = "2020-01-01"
DATE_TILL = "2026-09-01"


# =========================
# MOEX
# =========================

print("\n=== MOEX ===")

for ticker in TICKERS:

    print(f"\nЗагрузка {ticker}...")

    df = load_data_moex(
        ticker=ticker,
        date_from=DATE_FROM,
        date_till=DATE_TILL
    )

    save_to_raw(df)


# =========================
# CBR — ВАЛЮТЫ
# =========================

print("\n=== CBR: ВАЛЮТЫ ===")

for currency in CURRENCIES:

    print(f"\nЗагрузка {currency}...")

    df_currency = load_data_cbr(
        currency_code=currency,
        date_from=DATE_FROM,
        date_till=DATE_TILL
    )

    if not df_currency.empty:
        save_currency_to_raw(df_currency)


# =========================
# CBR — МЕТАЛЛЫ
# =========================

print("\n=== CBR: МЕТАЛЛЫ ===")

df_metals = load_data_metals(
    date_from=DATE_FROM,
    date_till=DATE_TILL
)

if not df_metals.empty:
    save_metals_to_raw(df_metals)


# =========================
# CBR — КЛЮЧЕВАЯ СТАВКА
# =========================

print("\n=== CBR: КЛЮЧЕВАЯ СТАВКА ===")

df_key_rate = load_data_key_rate(
    date_from=DATE_FROM,
    date_till=DATE_TILL
)

if not df_key_rate.empty:
    save_key_rate_to_raw(df_key_rate)