import numpy as np
import pandas as pd
import pickle
import glob
from datetime import datetime, timedelta
from retrieve_data import *
from onchain_data import retrieve_onchain_data

def generate_coin_list():
    csvfiles = []
    for file in glob.glob("raw_data/*.csv"):
        csvfiles.append(file)

    cryptolist = dict()
    for i in range(len(csvfiles)):
        df = pd.read_csv(csvfiles[i])
        df = df.iloc[:50,:]
        crypto_list = []
        for j in range(50):
            price = float(df['Price'].iloc[j][1:].replace(',',''))
            if abs(price - 1) < 0.05:
                continue
            crypto_list.append(df['Symbol'].iloc[j])
            if len(crypto_list) == 20:
                break
        name = csvfiles[i][9:-4]
        cryptolist[name] = crypto_list
    print(cryptolist)
    with open('coin_list','wb') as file:
        pickle.dump(cryptolist, file)

# print(coin_dict)

def update_coin_list(backtest_time, trade_period, coin_dict):
    
    trade_start_date_list = [backtest_time[i] for i in range(0, len(backtest_time), trade_period)]
    coin_change_date_list = list(coin_dict.keys())
    coin_change_date_list = pd.to_datetime(coin_change_date_list, format = '%Y%m%d').strftime('%Y-%m-%d %H:%M:%S')
    updated_coin_dict = dict()
    for i in range(len(trade_start_date_list)):
        date = trade_start_date_list[i]
        
        index = next(x for x, val in enumerate(coin_change_date_list) if val >= date)
        coin_change_date = coin_change_date_list[index-1]
        
        original_key = coin_change_date[:10].replace('-','')
        updated_coin_dict[date] = coin_dict[original_key]
    return updated_coin_dict



def fetch_data(coin_dict, buffer_period, train_period, trade_period, onchain_metrics_list):
    total_data = dict()
    for start_trade_date in list(coin_dict.keys()):
        start_trade_date_ = datetime.strptime(start_trade_date, '%Y-%m-%d %H:%M:%S')
        start_date_ = start_trade_date_ - timedelta(hours = (buffer_period + train_period))
        start_time = start_date_.timestamp()

        end_date_ = start_trade_date_ + timedelta(hours = (trade_period))
        if end_date_ > datetime.now():
            end_date_ = datetime.now()
        end_time = end_date_.timestamp()
        temp_coin_list = coin_dict[start_trade_date]
        temp_total_data = dict()
        print(start_time)
        on_chain_data = retrieve_onchain_data(onchain_metrics_list, int(start_time), int(end_time))
        for i in range(len(temp_coin_list)):
            print(temp_coin_list[i])
            try:
                temp_data = Future(temp_coin_list[i], resolution=3600).get_candlesticks(start_time, end_time)
                
                temp_data = temp_data.merge(on_chain_data, how = 'left', right_index = True, left_index = True)
                print(temp_data)
                temp_total_data[temp_coin_list[i]] = temp_data
            except:
                print('error!')
            
            
    #     total_data[start_trade_date] = temp_total_data
    # with open('total_data.pickle', 'wb') as file:
    #     pickle.dump(total_data, file)



if __name__ == '__main__':
    onchain_metrics_list = ['gas_price', 'rhodl_ratio', 'cvdd', 'nvts', 'unrealized_profit', 'ssr_oscillator']
    backtest_start_time = '2021-05-01'
    backtest_end_time = '2022-05-01'
    backtest_time = pd.date_range(backtest_start_time, backtest_end_time, freq = 'H').strftime('%Y-%m-%d %H:%M:%S')

    buffer_period = 10 * 24 
    train_period = 6 * 30 * 24
    trade_period = 2 * 30 * 24
    with open('coin_list', 'rb') as file:
        coin_dict = pickle.load(file)
    coin_dict = update_coin_list(backtest_time, trade_period, coin_dict)
    fetch_data(coin_dict, buffer_period, train_period, trade_period, onchain_metrics_list)
