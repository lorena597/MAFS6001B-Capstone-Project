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
                if args == 'skip this strat!':
                    continue
                elif isinstance(args, Iterable):
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

  
    def forward_chaining(self, forward_data: dict, K: int, params_range: dict):

        # for forward_data, key of dict represents different crypto
        # for params_range, key of dict represents different strats

        crypto_strats_and_params = {}

        for crypto in self.crypto_list:

            k_list = list(np.arange(K))
            strats_params = {}
            strats_sr = pd.DataFrame(np.nan, index = self.strategy_list, columns = k_list)
            
            fwd = forward_data[crypto]
            fwd['ix'] = np.arange(len(fwd))
            fwd['group'] = pd.cut(fwd['ix'], K, labels = k_list)

            for strats in self.strategy_list:

                sr_mat = np.full((K,K),np.nan)
                params_all = []
                
                for k1 in k_list:
                    train = fwd[fwd['group'] == k1]
                    args = [[train, params_range[strats], strats]]
                    with Pool(4) as pool:
                        params_k1 = pool.starmap(get_optimal_params, args)
                    params_all.append(params_k1)
                    for k2 in k_list:
                        if k2 == k1: continue
                        test = fwd[fwd['group'] == k2]
                        performance = get_performance(params_k1, Strategy(test), strats, self.transaction_bps, self.stoploss)
                        # ret_mat[k1,k2] = performance.iloc[0,0] 
                        # mdd_mat[k1,k2] = performance.iloc[0,2] 
                        sr_mat[k1,k2] = performance.iloc[0,3]

                # ret_avg = np.nanmean(ret_mat, axis = 1)
                # mdd_filter = np.any(mdd_mat <= -0.1, axis = 1)

                # ret_avg[mdd_filter == True] = - np.inf
                # if np.all(ret_avg, -np.inf):
                #    strats_candidate[strats] = 'skip this strat!'

                sr_avg = np.nanmean(sr_mat, axis = 1)
                max_ix = np.argmax(sr_avg)
                optimal_params = params_all[max_ix]
                strats_params[strats] = optimal_params
                strats_sr.loc[strats,:] = sr_mat[np.argmax(sr_avg),:]

            strats_sr_avg = np.nanmean(strats_sr, axis = 1)
            sorted_ix = np.argsort(strats_sr_avg)

            selected_strats_and_params = {}
            for ix in sorted_ix[-5:]:
                selected_strats = self.strategy_list[ix]
                selected_strats_and_params[selected_strats] = strats_params[selected_strats]
        
            crypto_strats_and_params[crypto] = selected_strats_and_params