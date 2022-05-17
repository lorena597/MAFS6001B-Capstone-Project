# from datetime import datetime
import datetime
import requests
import pandas as pd
from dateutil import tz

def retrieve_onchain_data(required_metrics_list, start_time, end_time):
    # insert your API key here
    API = 'https://api.glassnode.com'
    API_KEY = '29ErB0DmBl41kWyl4K6AnAvyiGk'
    to_zone = tz.tzlocal()
    result_df = pd.DataFrame()
    for metrics in required_metrics_list:
        # make API request
        if metrics == 'gas_price':
            res = requests.get(API + '/v1/metrics/fees/gas_price_mean', params={'a': 'ETH', 's': start_time, 'u': end_time+3600, 'i': '1h', 'api_key': API_KEY})
        else:
            res = requests.get(API + '/v1/metrics/indicators/' + metrics, params={'a': 'BTC', 's': start_time, 'u': end_time+3600, 'i': '24h', 'api_key': API_KEY})
        df = pd.read_json(res.text, convert_dates=['t'])
        df.rename(columns={'v': metrics}, inplace=True)
        df['t'] = df['t'].dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong').dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df.set_index('t')
        df = df.rename_axis(index=None)
        result_df = pd.concat([result_df, df], axis=1, join='outer')
    result_df.iloc[0:25,:] = result_df.iloc[0:25,:].fillna(method='bfill')
    result_df.fillna(method='ffill', inplace=True)
    result_df = result_df.iloc[:-1, :]

    return result_df


# if __name__ == '__main__':
#     onchain_metrics_list = ['gas_price', 'rhodl_ratio', 'cvdd', 'nvts', 'unrealized_profit', 'ssr_oscillator']
#     start_time = int(datetime.datetime(2020, 10, 23).timestamp())
#     end_time   = int(datetime.datetime(2021, 6, 30).timestamp())
#     data = retrieve_onchain_data(onchain_metrics_list, start_time, end_time)
