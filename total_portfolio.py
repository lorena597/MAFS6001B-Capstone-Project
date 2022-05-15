import itertools
import numpy as np
import pandas as pd
from strategy import Strategy
from performance import PerformanceAnalysis
from trade import Trade
from optimizer import *
from multiprocessing import Pool
from collections.abc import Iterable

def get_performance(params, strategy_: Strategy, current_strategy: str, transaction_bps: int, stoploss: int):
    if isinstance(params, Iterable):
        res = getattr(strategy_, current_strategy)(*params)
    else:
        res = getattr(strategy_, current_strategy)(params)
    
    trade_ = Trade(res, transaction_bps, stoploss)
    res = trade_.backtest()
    name = params
    performance = PerformanceAnalysis(res, name, trade_.winrate)
    return performance

def get_optimal_params(train: pd.DataFrame, params_range: list, current_strategy: str, bps: int, stoploss: int):
    strategy_ = Strategy(train)
    performance = pd.DataFrame()
    if len(params_range) == 0: return None
    if isinstance(params_range, Iterable):
        total_params = list(itertools.product(params_range))
    else:
        total_params = list(itertools.product(*params_range))

    for temp_params in total_params:
        temp_performance = get_performance(temp_params, strategy_, current_strategy, bps, stoploss)
        performance = pd.concat([performance, temp_performance.describe()])
    performance['rank'] = performance['Sharpe Ratio'].rank()
    chosen_params = performance.index[performance['rank'].argmax()]
    return chosen_params

class Total_portfolio():

    def __init__(self, bps: int, stoploss: int, crypto_dict: dict, strategy_list: list, params_range: list, selected_optimizer: list):
        
        self.transaction_bps = bps
        self.stoploss = stoploss
        self.crypto_dict = crypto_dict 
        self.strategy_list = strategy_list
        self.params_range = params_range
        self.selected_optimizer = selected_optimizer
        self.portfolio_return_dict = {}
        for i in range(len(self.selected_optimizer)):
            self.portfolio_return_dict[i] = []

    def combine(self, trade_data_dict: dict, crypto_strats_and_params, optimizer_ix):

        selected_optimizer = self.selected_optimizer[optimizer_ix]
        crypto_returns = pd.DataFrame()
        # save returns for each crypto

        for k in trade_data_dict.keys():
            crypto_data = trade_data_dict[k]
            selected_strategy_list = list(crypto_strats_and_params[k].keys())
            selected_params_list = list(crypto_strats_and_params[k].values())
            strategy_ = Strategy(crypto_data)
            strats_returns = pd.DataFrame()
            for i in range(len(selected_strategy_list)):
                args = selected_params_list[i]
                if isinstance(args, Iterable):
                    temp_res = getattr(strategy_, selected_strategy_list[i])(*args)
                else:
                    temp_res = getattr(strategy_, selected_strategy_list[i])(args)
                trade_ = Trade(temp_res, self.transaction_bps, self.stoploss)
                temp_res = trade_.backtest()
                strats_returns[i] = temp_res.portfolio_return
            
            mu_s = strats_returns.mean().values.reshape((-1,1))
            sigma_s = strats_returns.cov().values
            weight_s = selected_optimizer(mu_s, sigma_s)

            crypto_returns[k] = strats_returns.values @ weight_s

        mu_c = crypto_returns.mean().values.reshape((-1,1))
        sigma_c = crypto_returns.cov().values
        weight_c = selected_optimizer(mu_c, sigma_c)

        portfolio_returns = crypto_returns.values @ weight_c
        
        self.portfolio_return_dict[optimizer_ix].extend(list(portfolio_returns))
  
    def forward(self, forward_data: dict, K: int) -> dict:

        crypto_strats_and_params = {}
        # save selected strats and params for each crypto currency
        # key 1: crypto; key2: strats; value: params

        for crypto in self.crypto_dict.keys():

            k_list = list(np.arange(K))
            strats_params = {}
            # save optimal params for each strats
            strats_sr = pd.DataFrame(np.nan, index = self.strategy_list, columns = k_list)
            # save sharpe ratio in each sub period for each strats with optimal params
            
            # spilt fwd to k folders
            fwd = forward_data[crypto]
            fwd['ix'] = np.arange(len(fwd))
            fwd['group'] = pd.cut(fwd['ix'], K, labels = k_list)

            for strats in self.strategy_list:

                sr_mat = np.full((K,K),np.nan)
                params_all = []
                
                for k1 in k_list:
                    train = fwd[fwd['group'] == k1]
                    args = [train, self.params_range[strats], strats, self.transaction_bps, self.stoploss]
                    params_k1 = get_optimal_params(*args)
                    params_all.append(params_k1)
                    for k2 in k_list:
                        if k2 == k1: continue
                        test = fwd[fwd['group'] == k2]
                        performance = get_performance(params_k1, Strategy(test), strats, self.transaction_bps, self.stoploss)
                        sr_mat[k1,k2] = performance.describe().iloc[0,3]

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
        return crypto_strats_and_params
        
    def get_portfolio(self):

        buffer_period = 10 * 24 
        train_period = 6 * 30 * 24
        trade_period = 2 * 30 * 24 

        dts = list(self.crypto_dict.values())[0].index 

        buffer_start_date_list = [i for i in range(0, len(dts), trade_period)]
        train_start_date_list = [i for i in range(buffer_period, len(dts), trade_period)]
        trade_start_date_list = [i for i in range(buffer_period+train_period, len(dts), trade_period)]

        rebalance_times = min(len(buffer_start_date_list), len(train_start_date_list), len(trade_start_date_list))
        buffer_start_date_list = buffer_start_date_list[:rebalance_times]
        train_start_date_list = train_start_date_list[:rebalance_times]
        trade_start_date_list = trade_start_date_list[:rebalance_times]

        for i in range(rebalance_times):
            forward_data_dict = {}
            for k in self.crypto_dict.keys():
                temp_df = self.crypto_dict[k].iloc[buffer_start_date_list[i]:trade_start_date_list[i],:]
                temp_df['not_buffer'] = 1
                temp_df['not_buffer'].iloc[buffer_start_date_list[i]:train_start_date_list[i]] = 0
                forward_data_dict[k] = temp_df
            crypto_strats_and_params = self.forward(forward_data_dict, 6)
            trade_data_dict = {}
            for k in self.crypto_dict.keys():
                if trade_start_date_list[i] + trade_period >= len(dts):
                    temp_df = self.crypto_dict[k].iloc[trade_start_date_list[i]-buffer_period:]
                else:
                    temp_df = self.crypto_dict[k].iloc[trade_start_date_list[i]-buffer_period:trade_start_date_list[i]+train_period]
                temp_df['not_buffer'] = 1
                temp_df['not_buffer'].iloc[0:buffer_period] = 0
                trade_data_dict[k] = temp_df
            for optimizer_ix in range(len(self.selected_optimizer)):
                self.combine(trade_data_dict, crypto_strats_and_params, optimizer_ix)
            print(f'{i} have finished!')
        return self.portfolio_return_dict
            
            

            
