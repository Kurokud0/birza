import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Компании для парса
tickers = ["SBER", "GAZP", "LKOH", "GMKN", "YDEX"]


def load_data_moex(ticker, data_from, data_till):

    moexurl = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"

    full_data = []
    iter_start = 0

    while True:

        params = {
            "from": data_from,
            "till": data_till,
            "interval": 24,
            "start": iter_start,
        }

        response = requests.get(moexurl, params=params)
        response.raise_for_status()
        candles = response.json()["candles"]
        rows = candles["data"]

        if not rows:
            break

        df_part = pd.DataFrame(
            rows,
            columns=candles["columns"]
        )

        full_data.append(df_part)
        print(
            f"{ticker}: загружено "
            f"{sum(len(x) for x in full_data)} строк"
        )

        iter_start += len(rows)


    df = pd.concat(full_data, ignore_index=True)
    df["ticker"] = ticker

    return df


data = []

for ticker in tickers:

    df_ticker = load_data_moex(
        ticker,
        "2020-01-01",
        "2026-09-01"
    )

    data.append(df_ticker)

df = pd.concat(data, ignore_index=True)

print(df.groupby("ticker").size())

##EDA

#print(df.info())
#print(df.shape)
#print(df.describe())

print(df.isna().sum())
df = df.dropna()

df = df.drop_duplicates()

print(df[df["ticker"] == "SBER"])

