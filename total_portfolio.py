import itertools
import numpy as np
import pandas as pd
from strategy import Strategy
from performance import PerformanceAnalysis
from trade import Trade
from multiprocessing import Pool
from collections.abc import Iterable

def get_performance(params, strategy_: Strategy, current_strategy: str, transaction_bps: int, stoploss: int):
    if isinstance(params, Iterable):
            res = getattr(strategy_, current_strategy)(*params)
    else:
        res = getattr(strategy_, current_strategy)(params)
    
    trade_ = Trade(res, transaction_bps, stoploss)
    res = trade_.backtest()
    name = current_strategy
    performance = PerformanceAnalysis(res, name)
    return performance

def get_optimal_params(train: pd.DataFrame, params_range: list, current_strategy: str):
    strategy_ = Strategy(train)
    prediction_accuracy = []
    params = []
    train['true_return'] = train.close.diff()
    true_position = np.sign(train[train['not_buffer'] == 1]['true_return'])

    if len(params_range) == 1:
        total_params = params_range[0]
    else:
        total_params = list(itertools.product(*params_range))

    for temp_params in total_params:
        if isinstance(temp_params, Iterable):
            res = getattr(strategy_, current_strategy)(*temp_params)
        else:
            res = getattr(strategy_, current_strategy)(temp_params)
        temp_train_position = np.array(res.position.shift().fillna(0))
        temp_accuracy = np.mean(temp_train_position == true_position)
        prediction_accuracy.append(temp_accuracy)
        params.append(temp_params)
    chosen_params = params[np.argmax(prediction_accuracy)]
    return chosen_params


class Total_portfolio():

    def __init__(self, bps: int, stoploss: int, crypto_list: list, params_range: dict):
        
        self.transaction_bps = bps
        self.stoploss = stoploss
        self.crypto_list = crypto_list
        self.strategy_list = [func for func in dir(Strategy) if not func.startswith('__')]
        self.rolling_return = []
        self.params_range = params_range
        self.params = dict()


    def get_portfolio(self, backtest_len: int, data: dict, selected_strategy: dict, params: dict, weight: dict):
        total_return = np.zeros(backtest_len)
        for i in range(self.crypto_list):
            strategy_ = Strategy(data[self.crypto_list[i]])
            temp_strategy = selected_strategy[self.crypto_list[i]]
            temp_params = params[self.crypto_list[i]]
            temp_weight = weight[self.crypto_list[i]]
            for j in range(temp_strategy):
                args = temp_params[j]
                temp_strategy_ = getattr(strategy_, temp_strategy[j])(*args)
                temp_res = temp_strategy_(strategy_.df)
                # need to modify temp_res because exists nan
                trade_ = Trade(temp_res, self.transaction_bps, self.stoploss)
                trade_.backtest()
                total_return += temp_weight[j] * np.array(trade_.portfolio_return_hist)

        self.rolling_return = self.rolling_return.extend(list(total_return))

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
        
                

