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
        'crossover': [list(range(50,100,10)), list(range(100,150,10))],
        'channelbreakout': [list(range(40,140,10)), [1.5,2,2.5]],
        'RSI': [list(range(3,20,3)), [10,20,30]],
        'ADX': [list(range(2,10,3)), [30,40,50]],
        'FR' : [list(range(1,30,5))],
        'RHODL': [list(range(1,7,3))],
        'CVDD': [list(range(1,24,3)), list(range(24,28,3))],
        'NVTS': [list(range(1,90,10))],
        'RUP': [list(range(1,24,3))],
        'SSRO': [list(range(1,24,3))]
}

strategy_list = list(params_range.keys())
selected_optimizer = [MSRP, MDCP]
port = Total_portfolio(15, 10, strategy_list, params_range, selected_optimizer)
res_dict = port.get_portfolio(total_data)
print('end!')
