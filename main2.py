import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy import Strategy
from trade import Trade
from performance import PerformanceAnalysis
from total_portfolio import *

with open('total_data.pickle', 'rb') as file:
        total_data = pickle.load(file)

params_range = {
        'RSI': [list(range(3,20)), [10,20,30]],
        'ADX': [list(range(2,10)), [30,40,50]],
        'FR' : [list(range(1,30))],
        'RHODL': [list(range(1,7))],
        'CVDD': [list(range(1,24)), list(range(24,28))],
        'NVTS': [list(range(1,90))],
        'RUP': [list(range(1,24))],
        'SSRO': [list(range(1,24))]
}

strategy_list = list(params_range.keys())
selected_optimizer = [IVP, EWP, MVP, GMVP, MSRP, MDP, MDCP]
port = Total_portfolio(15, 10, strategy_list, params_range, selected_optimizer)
res_dict = port.get_portfolio(total_data)
print('end!')
