import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

moexurl = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/SBER/candles.json"

params =  {
    "from": "2025-01-01",
    "till": "2026-09-01",
    "interval": 24
}

response = requests.get(moexurl, params=params)
response.raise_for_status()

data = response.json()["candles"]

df = pd.DataFrame(
    data["data"],
    columns=data["columns"]
)

##EDA

#print(df.info())
#print(df.shape)
#print(df.describe())

if df.count().isnull().sum() > 0:
    df_cleaned = df.fillna(df.mean())
df_duplicates = df.drop_duplicates()

