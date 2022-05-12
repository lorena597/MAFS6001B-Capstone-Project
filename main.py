import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy import Strategy
from trade import Trade
from performance import PerformanceAnalysis

'''Data Preparation'''

df = pd.read_csv(r"C:\Users\loren\Documents\HKUST\Capstone Project\final\ETHUSDT_1h.csv")

cols = ["open time", "open", "high", "low",
        "close", "volume", "close time", "quote asset volume",
        "number of trades", "taker buy volume", "taker buy quote asset volume", "ignore"]
df.columns = cols

df["open time"] = df["open time"] // 1000
df.index = [datetime.datetime.utcfromtimestamp(df["open time"][i]) for i in range(len(df))]

'''Example'''

train = df[("2021-07-01 00:00:00" <= df.index) & (df.index <= "2022-01-01 00:00:00")]
test = df[("2022-01-01 00:00:00" < df.index) & (df.index <= "2022-03-01 00:00:00")]

strats = Strategy(train)
res = Trade(strats.ADX(7)).trade()
performance = PerformanceAnalysis(res, 'NAME').describe()

'''Validation'''
whole = pd.concat([train, test])
strats = Strategy(whole)
train_performance = pd.DataFrame()
test_performance = pd.DataFrame()
for i in range(2,5):
        res = Trade(strats.ADX(i)).trade()
        name = 'ADX {}'.format(i)
        train_performance = pd.concat([train_performance, PerformanceAnalysis(res[:len(train)], name).describe()])
        test_performance = pd.concat([test_performance, PerformanceAnalysis(res[-len(test):], name).describe()])
rank = train_performance['Sharpe Ratio'].rank()
test_performance.iloc[rank.argmax(), :]


        






