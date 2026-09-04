import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_moex(ticker, data_from, data_till):
    moexurl = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"

    params = {
        "from": data_from,
        "till": data_till,
        "interval": 24
    }

    response = requests.get(moexurl, params=params)
    response.raise_for_status()

    data = response.json()["candles"]

    df = pd.DataFrame(
        data["data"],
        columns=data["columns"]
    )

    df["ticker"] = ticker

    return df

tickers = ["SBER", "GAZP", "LKOH", "GMKN", "YDEX"]

full_data = []

for ticker in tickers:
    df_ticker = load_data_moex(ticker, "2025-01-01", "2026-09-04")
    full_data.append(df_ticker)

    df = pd.concat(full_data)


##EDA

#print(df.info())
#print(df.shape)
#print(df.describe())

df = df.dropna()
df = df.drop_duplicates()

print(df.groupby("ticker").size())
print(df)

