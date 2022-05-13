import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy import Strategy
from trade import Trade
from performance import PerformanceAnalysis

'''Data Preparation'''

df = pd.read_csv(r"C:\Users\loren\Documents\HKUST\Capstone Project\final\MAFS6001B-Capstone-Project\ETHUSDT_1h.csv")

cols = ["open time", "open", "high", "low",
        "close", "volume", "close time", "quote asset volume",
        "number of trades", "taker buy volume", "taker buy quote asset volume", "ignore"]
df.columns = cols

df["open time"] = df["open time"] // 1000
df.index = [datetime.datetime.utcfromtimestamp(df["open time"][i]) for i in range(len(df))]

'''Example'''

train = df[("2021-07-01 00:00:00" <= df.index) & (df.index <= "2022-01-01 00:00:00")]
test = df[("2022-01-01 00:00:00" < df.index) & (df.index <= "2022-03-01 00:00:00")]
start_date = train.index[0]
end_date = train.index[-1]

strats = Strategy(df, start_date, end_date, 200)
res = Trade(strats.RSI(40), 15, 10).backtest()
performance = PerformanceAnalysis(res, 'NAME').describe()

print('end')