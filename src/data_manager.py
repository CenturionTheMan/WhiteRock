import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import tabulate as tb


TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = None
TIMEFRAME = "1d"    # 1d, 1h, 15m, 5m
RANDOM_SEED = 42


df = yf.download(
    TICKER, 
    start=START_DATE,
    end=END_DATE,
    interval=TIMEFRAME,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

df.dropna(inplace=True)

print(tb.tabulate(df.tail(5), headers='keys', tablefmt='psql'))

df["rsi"] = ta.rsi(df["Close"], length=14)
macd = ta.macd(df["Close"])
df = df.join(macd)

bbands = ta.bbands(df["Close"], length=20)
df = df.join(bbands)

df["ema_10"] = ta.ema(df["Close"], length=10)
df["ema_20"] = ta.ema(df["Close"], length=20)
df["ema_50"] = ta.ema(df["Close"], length=50)
df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
stoch = ta.stoch(df["High"], df["Low"], df["Close"])
df["stoch_k"] = stoch.iloc[:, 0]
df["stoch_d"] = stoch.iloc[:, 1]
df["returns"] = df["Close"].pct_change()

df.dropna(inplace=True)

print(tb.tabulate(df.tail(5), headers='keys', tablefmt='psql'))

df.to_csv(f"data/{TICKER}.csv")