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
strategy_list = ['POS','WMA','VR','RSI']
ls = [24,48]
params_range = {'POS': ls, 'WMA': ls, 'VR': ls, 'RSI': ls}
selected_optimizer = [IVP, EWP]
port = Total_portfolio(15, 10, strategy_list, params_range, selected_optimizer)
res_dict = port.get_portfolio(total_data)
print('end!')