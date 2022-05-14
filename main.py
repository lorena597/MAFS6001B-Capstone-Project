import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy import Strategy
from trade import Trade
from performance import PerformanceAnalysis
from total_portfolio import *

'''Data Preparation'''

df = pd.read_csv(r"C:\Users\loren\Documents\HKUST\Capstone Project\final\MAFS6001B-Capstone-Project\ETHUSDT_1h.csv")

cols = ["open time", "open", "high", "low",
        "close", "volume", "close time", "quote asset volume",
        "number of trades", "taker buy volume", "taker buy quote asset volume", "ignore"]
df.columns = cols

df["open time"] = df["open time"] // 1000
df.index = [datetime.datetime.utcfromtimestamp(df["open time"][i]) for i in range(len(df))]

# 不同coin的df的index应该是一致的不然可能会对后面有影响
# 如果不一样就index取交集处理一下
crypto_dict = {'A': df, "B": df}
strategy_list = ['POS','WMA','VR','OBV','RSI']
ls = [24,48]
params_range = {'POS': ls, 'WMA': ls, 'OBV': ls, 'VR': ls, 'RSI': ls}
selected_optimizer = [IVP, EWP]
port = Total_portfolio(15, 10, crypto_dict, strategy_list, params_range, selected_optimizer)
res_dict = port.get_portfolio()
print('end!')