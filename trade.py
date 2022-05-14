from turtle import reset
import numpy as np
import pandas as pd
import logging
from strategy import Strategy
from performance import PerformanceAnalysis

class Trade(object):
    
    def __init__(self, res: pd.DataFrame, bps: int, stoploss: int, if_log = False):
        self.res = res
        self.transaction_perc = bps * 0.0001 # transaction cost
        self.stoploss = stoploss # stop loss
        self.position = 0
        self.winrate = 0
        self.trade_num = 0
        self.win_trade_num = 0
        self.portfolio_return = 0
        self.start_trade_value = 0
        self.portfolio_value = 1
        self.portfolio_return_hist = []
        self.portfolio_value_hist = []
        self.if_log = False
        if if_log:
            self.init_logging('backtest')

    def init_logging(self, filename: str):
        file = open(f"{filename}.log", "w")
        file.close()
        logging.basicConfig(level = logging.INFO, filename = f'{filename}.log', filemode = 'a', format = '%(message)s')
        self.log = logging

    def entry_position(self, i: int):
    
        if self.res.entry_signal[i] == 1:
            temp_position = 1
            if self.if_log:
                self.log.info(f'{self.res.index[i]} long')
        elif self.res.entry_signal[i] == -1:
            temp_position = -1
            if self.if_log:
                self.log.info(f'{self.res.index[i]} short')
        transaction_cost = - self.transaction_perc * abs(self.position - temp_position)
        self.portfolio_return += transaction_cost
        self.position = temp_position
        self.start_trade_value = self.portfolio_value
        
    def exit_position(self, i: int):

        temp_position = 0
        if self.if_log:
            self.log.info(f'{self.res.index[i]} exit')
        transaction_cost = - self.transaction_perc * abs(self.position - temp_position)
        self.portfolio_return += transaction_cost
        self.position = temp_position
        self.trade_num += 1
        if self.portfolio_value > self.start_trade_value:
            self.win_trade_num += 1
        self.start_trade_value = 0

    def stop_loss(self, i: int):
        if self.position != 0:
            current_trade_loss = (self.portfolio_value - self.start_trade_value) / self.start_trade_value * 100
            if current_trade_loss <= - self.stoploss:
                self.exit_position(i)

    def mark_to_market(self, i: int):

        if i >= 1:
            self.portfolio_return = (self.res.close[i] / self.res.close[i-1] - 1) * self.position
            self.portfolio_value = self.portfolio_value * (1 + self.portfolio_return)
        self.portfolio_return_hist.append(self.portfolio_return)
        self.portfolio_value_hist.append(self.portfolio_value)
    
    def backtest(self):
        
        for i in range(len(self.res)):
            self.mark_to_market(i)
            self.stop_loss(i)
            if self.res.entry_signal[i] != 0:
                self.entry_position(i)
                
            elif self.res.exit_signal[i] == 1 and self.position != 0:
                self.exit_position(i)
        
        self.res['portfolio_return'] = self.portfolio_return_hist
        self.res['portfolio_value'] = self.portfolio_value_hist
        self.winrate = self.win_trade_num/self.trade_num*100
        return self.res
        

