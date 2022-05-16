import pandas as pd
import requests
from datetime import datetime
from dateutil import parser


def _get(
        endpoint,
        params,
        api="https://ftx.com/api"
):

    try:
        response = requests.get(api+endpoint, params=params)
        response.raise_for_status()
    except Exception as err:
        print(err)
    else:
        return response.json()


def get_candlesticks(
        market,
        start_time=None,
        end_time=None,
        limit=1500,
        resolution=3600,
        close_only=False,
):

    results = []
    while True:
        result = _get(f'/markets/{market}/candles', params={
            'start_time': start_time,
            'end_time': end_time,
            'resolution': resolution,
            'limit': limit,
        })['result']
        results += result
        if 0 <= len(result) < limit:
            break
        end_time = min(t['time'] for t in result)//1000

    df = pd.DataFrame(results)
    df['time'] = pd.to_datetime(df['startTime']).dt.tz_convert("Asia/Hong_Kong").dt.strftime('%Y-%m-%d %H:%M:%S')
    df = (df.drop_duplicates(subset='time', keep='last')
          .set_index('time')
          .sort_index()
          .drop('startTime', axis=1))

    if close_only:
        df = df.drop(['open', 'high', 'low', 'volume'], axis=1)

    return df


def get_funding_rates(
        future,
        start_time=None,
        end_time=None,
        limit=500,
):

    results = []
    while True:
        result = _get(f'/funding_rates', params={
            'start_time': start_time,
            'end_time': end_time,
            "future": future,
        })['result']
        results += result
        if 0 <= len(result) < limit:
            break
        end_time = min(parser.parse(t['time']) for t in result).timestamp()

    df = pd.DataFrame(results)
    df['rate'] = df['rate'].apply(float)
    df['time'] = pd.to_datetime(df['time']).dt.tz_convert("Asia/Hong_Kong").dt.strftime('%Y-%m-%d %H:%M:%S')
    df = (df.drop_duplicates(subset='time', keep='last')
          .set_index('time')
          .sort_index()
          .drop('future', axis=1))

    return df


class Spot:
    def __init__(self, ticker, resolution=3600, suffix='/USD') -> None:
        self.ticker = ticker
        self.suffix = suffix
        self.resolution = resolution

    def get_candlesticks(self, start=None, end=None, close_only=False):
        return get_candlesticks(self.ticker+self.suffix, start, end, resolution=self.resolution, close_only=close_only)


class Future(Spot):
    def __init__(self, ticker, resolution=3600, suffix='-PERP') -> None:
        super().__init__(ticker, resolution)
        self.suffix = suffix

    def get_funding_rates(self, start=None, end=None):
        return get_funding_rates(self.ticker+self.suffix, start, end)
