from datetime import date

from moex import (
    load_data_moex,
    save_to_raw,
    load_data_index,
    save_index_to_raw
)

from cbr import (
    load_data_cbr,
    load_data_metals,
    load_data_key_rate,
    load_data_zcyc,
    save_currency_to_raw,
    save_metals_to_raw,
    save_key_rate_to_raw,
    save_zcyc_to_raw
)


TICKERS = [
    "SBER", "GAZP", "LKOH", "GMKN", "YDEX",
    "ROSN", "NVTK", "TATN", "MGNT", "MTSS",
    "MOEX", "VTBR", "AFLT", "ALRS", "CHMF",
    "NLMK", "PLZL", "PHOR", "SNGS", "SNGSP",
    "TCSG", "OZON", "RUAL", "PIKK", "IRAO"
]

CURRENCIES = [
    "USD", "EUR", "CNY", "GBP", "JPY",
    "CHF", "TRY", "AED", "KZT", "INR"
]

INDICES = ["IMOEX"]

DATE_FROM = "2020-01-01"
DATE_TILL = date.today().isoformat()


def main():
    print("ПОЛНАЯ ЗАГРУЗКА ДАННЫХ")

    print("\nMOEX АКЦИИ")

    for ticker in TICKERS:
        print(f"{ticker}...")

        df = load_data_moex(
            ticker,
            DATE_FROM,
            DATE_TILL
        )

        if not df.empty:
            save_to_raw(df)

    print("\nMOEX ИНДЕКСЫ")

    for index_code in INDICES:
        print(f"{index_code}...")

        df = load_data_index(
            index_code,
            DATE_FROM,
            DATE_TILL
        )

        if not df.empty:
            save_index_to_raw(df)

    print("\nCBR ВАЛЮТЫ")

    for currency in CURRENCIES:
        print(f"{currency}...")

        df = load_data_cbr(
            currency,
            DATE_FROM,
            DATE_TILL
        )

        if not df.empty:
            save_currency_to_raw(df)

    print("\nCBR МЕТАЛЛЫ")

    df = load_data_metals(
        DATE_FROM,
        DATE_TILL
    )

    if not df.empty:
        save_metals_to_raw(df)

    print("\nCBR КЛЮЧЕВАЯ СТАВКА")

    df = load_data_key_rate(
        DATE_FROM,
        DATE_TILL
    )

    if not df.empty:
        save_key_rate_to_raw(df)

    print("\nCBR ZCYC")

    df = load_data_zcyc(
        DATE_FROM,
        DATE_TILL
    )

    if not df.empty:
        save_zcyc_to_raw(df)

    print("\nПОЛНАЯ ЗАГРУЗКА ЗАВЕРШЕНА")


if __name__ == "__main__":
    main()