import itertools
import numpy as np
import pandas as pd
from strategy import Strategy
from performance import PerformanceAnalysis
from trade import Trade
from optimizer import *
from multiprocessing import Pool
from collections.abc import Iterable

def get_performance(params, strategy_: Strategy, current_strategy: str, transaction_bps: int, stoploss: int, get_param_perf = False):
    if isinstance(params, Iterable):
            res = getattr(strategy_, current_strategy)(*params)
    else:
        res = getattr(strategy_, current_strategy)(params)
    
    trade_ = Trade(res, transaction_bps, stoploss)
    res = trade_.backtest()
    if get_param_perf:
        name = params
    else:
        name = current_strategy
    performance = PerformanceAnalysis(res, name, trade_.winrate)
    return performance

def get_optimal_params(train: pd.DataFrame, params_range: list, current_strategy: str, bps: int, stoploss: int):
    strategy_ = Strategy(train)
    performance = pd.DataFrame()
    if len(params_range) == 1:
        total_params = params_range[0]
    else:
        total_params = list(itertools.product(*params_range))

    for temp_params in total_params:
        temp_performance = get_performance(temp_params, strategy_, current_strategy, bps, stoploss, get_param_perf = True)
        performance = pd.concat([performance, temp_performance])
    performance['rank'] = performance['Sharpe Ratio'].rank()
    chosen_params = performance.index[performance['rank'].argmax()]
    return chosen_params

class Total_portfolio():

    def __init__(self, bps: int, stoploss: int, crypto_list: list, params_range: dict):
        
        self.transaction_bps = bps
        self.stoploss = stoploss
        self.crypto_list = crypto_list
        self.strategy_list = [func for func in dir(Strategy) if not func.startswith('__')]
        self.rolling_return = []
        self.rolling_crypto_returns = pd.DataFrame()
        self.params_range = params_range
        self.params = dict()


    # use selected strategies with corresponding parameters to backtest for each cryptocurrency
    # use selected optimizer to combine different strategies and different cryptocurrenies respectively
    def get_portfolio(self, data: dict, selected_strategy: dict, selected_params: dict, selected_optimizer: function):

        crypto_returns = pd.DataFrame()

        for i in self.crypto_list:
            current_crypto = self.crypto_list[i]
            selected_strategy_list = selected_strategy[current_crypto]
            if len(selected_strategy_list) == 0: continue
            strategy_ = Strategy(data[current_crypto])
            selected_params_list = selected_params[current_crypto]

            strats_returns = pd.DataFrame()
            
            for j in selected_strategy_list:
                args = selected_params_list[j]
                if isinstance(args, Iterable):
                    temp_res = getattr(strategy_, selected_strategy_list[j])(*args)
                else:
                    temp_res = getattr(strategy_, selected_strategy_list[j])(args)
                trade_ = Trade(temp_res, self.transaction_bps, self.stoploss)
                temp_res = trade_.backtest()
                strats_returns[j] = temp_res.portfolio_return
            
            mu_s = strats_returns.mean().values.reshape((-1,1))
            sigma_s = strats_returns.cov().values
            weight_s = selected_optimizer(mu_s, sigma_s)

            crypto_returns[current_crypto] = strats_returns.values @ weight_s

        mu_c = crypto_returns.mean().values.reshape((-1,1))
        sigma_c = crypto_returns.cov().values
        weight_c = selected_optimizer(mu_c, sigma_c)

        portfolio_returns = crypto_returns.values @ weight_c

        self.rolling_return = self.rolling_return.extend(list(portfolio_returns))
        self.rolling_crypto_returns = pd.concat([self.rolling_crypto_returns, crypto_returns])

    # Forward chaining is a cross validation method for time series  
    def Forward_chaining(self, params_range: dict, df: pd.DataFrame, K: int):
        
        df['ix'] = np.arange(len(df))
        df['group'] = pd.cut(df['ix'], K, lables = list(np.arange(K)))
        chosen_strategy = []
        for i in range(len(self.crypto_list)):
            performance = pd.DataFrame()
            for k in range(1, K):
                train = df[df['group'] < k]
                test = df[df['group'] == k]
                args = [[train, self.params_range[temp_strat], temp_strat, self.transaction_bps, self.stoploss] for temp_strat in self.strategy_list]
                with Pool(4) as pool:
                    chosen_params = pool.starmap(get_optimal_params, args)
                performance_ = pd.DataFrame()
                for j in range(len(chosen_params)):
                    temp_performance_ = get_performance(chosen_params[j], Strategy(test), self.strategy_list[j], self.transaction_bps, self.stoploss)
                    performance_ = pd.concat([performance_, temp_performance_])
                if len(performance) == 0:
                    performance = performance_
                else:
                    performance = performance + performance_
            performance = performance/K
