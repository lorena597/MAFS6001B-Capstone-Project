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
        'RSI': [48, 96],
        'ADX': [2, 6, 8]
}

strategy_list = list(params_range.keys())
selected_optimizer = [IVP, EWP, MVP, GMVP, MSRP, MDP, MDCP]
port = Total_portfolio(15, 10, strategy_list, params_range, selected_optimizer)
res_dict = port.get_portfolio(total_data)
print('end!')