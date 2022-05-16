from turtle import reset
import numpy as np
import pandas as pd
from sympy import N
# import talib

class Strategy(object):
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def init_res(self):
        res = pd.DataFrame(index = self.df.index, columns = ["close","indicator","position", "entry_signal","exit_signal","not_buffer"])
        res.close = self.df.close
        res.not_buffer = self.df.not_buffer
        return res

    def calculate_signal(self, res: pd.DataFrame) -> pd.DataFrame:
        raw_signal = res.position.diff()
        entry_long = (raw_signal > 0)*(res.position == 1).astype(int)
        entry_short = (raw_signal < 0)*(res.position == -1).astype(int)
        exit_signal = (raw_signal != 0)*(res.position == 0).astype(int)
        res.entry_signal = entry_long - entry_short
        res.exit_signal = exit_signal
        res = res[res['not_buffer'] == 1]
        res.reset_index(inplace = True, drop = True)
        return res

    def crossover(self, short: int, long: int) -> pd.DataFrame:
        res = self.init_res()
        res.indicator = self.df.close.rolling(short).mean() - self.df.close.rolling(long).mean()
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res
    
    def channelbreakout(self, window: int, threshold: float) -> pd.DataFrame:
        res = self.init_res()
        res.indicator = (self.df.close - self.df.close.rolling(window).mean()) / self.df.close.rolling(window).std()
        res.position = np.sign(res.indicator) * (abs(res.indicator) > threshold)
        res = self.calculate_signal(res)    
        return res
    
    def POS(self, window: int) -> pd.DataFrame:
        res = self.init_res()
        
        H = self.df.high.rolling(window).max().values
        L = self.df.low.rolling(window).min().values
        
        res.indicator = (res.close - L) / (H - L)*100
        long = (res.indicator > 80).astype(int)
        short = (res.indicator < 20).astype(int)
        res.position = long - short
        res = self.calculate_signal(res)
        
        return res
            
    def RSI(self, window: int, benchmark_percent: int) -> pd.DataFrame:
        res = self.init_res()
        
        change = self.df.close - self.df.close.shift()
        abs_sum = abs(change).rolling(window).sum()
        change[change < 0] = 0
        up_sum = change.rolling(window).sum()
        res.indicator = up_sum / abs_sum * 100
        
        long = (res.indicator < benchmark_percent).values.astype(int)
        short = (res.indicator > 100-benchmark_percent).values.astype(int)
        res.position = long - short
        res = self.calculate_signal(res)

        return res
    
    def OBV(self, window: int):
        res = self.init_res()
        
        sign = np.sign(self.df.close - self.df.close.shift())
        OBV = np.cumsum(self.df.volume * sign)
        res.indicator = OBV.pct_change(window)*100
        long = (res.indicator > 10)*(res.pct_change(window) > 10).astype(int)
        short = (res.indicator <= -10)*(res.pct_change(window) < -10).astype(int)
        res.position = long - short
        res = self.calculate_signal(res)

        return res
    
    def VR(self, window: int) -> pd.DataFrame:
        res = self.init_res()
        
        sign = np.sign(self.df.close - self.df.close.shift())
        
        AV = (sign == 1) * self.df.volume
        BV = (sign == -1) * self.df.volume
        CV = (sign == 0) * self.df.volume
        
        AVS = AV.rolling(window).sum()
        BVS = BV.rolling(window).sum()
        CVS = CV.rolling(window).sum()
        
        VR = (AVS + CVS / 2) / (BVS + CVS / 2) * 100
        MAVR = VR.rolling(window).mean()
        
        res.indicator = MAVR
        
        long = MAVR < 40
        short = MAVR > 150
        res.position = long.astype(int) - short.astype(int)
        res = self.calculate_signal(res)
        return res
    
    def ADX(self, window: int, benchmark_percent: int) -> pd.DataFrame:
        res = self.init_res()

        adx = talib(self.high, self.low, self.close, timeperiod = window)
        res.indicator = talib.DX(self.high, self.low, self.close, timeperiod = window)
        res.position = np.sign(res.indicator)*(adx > benchmark_percent)
        res = self.calculate_signal(res)
        return res
    
    def smooth(self, series: pd.Series, window: int) -> pd.Series:
        series.fillna(method = "ffill", inplace = True)
        for i in range(len(series)):
            if i == 0 or np.isnan(series[i-1]):
                continue
            else:
                series[i] = (series[i-1] * (window - 1) + series[i]) / window
        return series

    # indicator from Colin's report
    def EFMOM(self, short: int, long: int) -> pd.DataFrame:

        res = self.init_res()

        rs = self.df.close / self.df.close.shift(short) - 1
        rl = self.df.close / self.df.close.shift(long) - 1
        pc = self.df.close.pct_change()
        ss = abs(pc).rolling(short).sum()
        sl = abs(pc).rolling(long).sum()
        
        res.indicator = rs / ss - rl / sl
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res

    # indicator from Colin's report
    def REG(self, window: int) -> pd.DataFrame:

        res = self.init_res()

        X = np.ones((window, 2))
        X[:, 1] = np.arange(window) + 1

        def func(y):
            y[np.isnan(y)] = np.nanmean(y)
            return np.linalg.inv(X.T @ X) @ ( X.T @ y)
        
        for i in range(window, len(res)):
            y = res.close[i-window:i]
            b, a = func(y)
            y_hat = y * a + b
            res.indicator[i-1] = y[-1] / y_hat[-1] - 1
        
        long = res.indicator > 0.05
        short = res.indicator < -0.05
        res.position = long.astype(int) - short.astype(int)
        res = self.calculate_signal(res)
        return res

    def SPMOM(self, short: int, long: int) -> pd.DataFrame:

        res = self.init_res()

        ret = res.close.pct_change()
        sp_short = ret.rolling(short).mean() / ret.rolling(short).std()
        sp_long = ret.rolling(long).mean() / ret.rolling(long).std()

        res.indicator = sp_short - sp_long
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res
    
    # indicator from Colin's report
    def INVVOL(self, short: int, long: int) -> pd.DataFrame:

        res = self.init_res()

        ret = res.close.pct_change()
        siv = 1 / ret.rolling(short).std()
        liv = 1 / ret.rolling(long).std()
        res.indicator = siv - liv
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res

    def WMA(self, window: int) -> pd.DataFrame:
        res = self.init_res()

        weight = np.array(list(reversed(range(1, window + 1))))/(window * (window + 1)/2)
        res.indicator = res.close.rolling(window).apply(lambda x: weight.dot(x))
        res.position = (res.close > res.indicator).astype(int)
        res = self.calculate_signal(res)
        return res

    # indicator from Colin's report
    def DBCD(self, window1: int, window2: int, window3: int) -> pd.DataFrame:
        res = self.init_res()

        bias = (res.close - res.close.rolling(window1).mean())/res.close.rolling(window1).mean()*100
        res.indicator = self.smooth(bias.diff(window2), window3)
        long = (res.indicator > 0.05).astype(int)
        short = (res.indicator < -0.05).astype(int)
        res.position = long - short
        res = self.calculate_signal(res)
        return res

    # indicator from Talib considering volume
    def ADOSC(self, fastperiod: int, slowperiod: int) -> pd.DataFrame:
        res = self.init_res()

        adosc = talib.ADOSC(self.df.high, self.df.low, self.df.close, self.df.volume, fastperiod = fastperiod, slowperiod = slowperiod)
        res.indicator = adosc
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res

    def FR(self, window: int) -> pd.DataFrame:
        res = self.init_res()

        res.indicator = self.df.funding_rates.rolling(window).mean()
        res.position = np.sign(-res.indicator)
        res = self.calculate_signal(res)

        return res

    def RHODL(self, window: int) -> pd.DataFrame:
        res = self.init_res()

        res.indicator = self.df.rhodl_ratio - self.df.rhodl_ratio.rolling(window).mean()
        res.position = np.sign(-res.indicator)
        res = self.calculate_signal(res)
        return res

    def CVDD(self, short:int, long:int) -> pd.DataFrame:
        res = self.init_res()

        res.indicator = self.df.cvdd.rolling(short).mean() - self.df.cvdd.rolling(long).mean()
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res

    def NVTS(self, window:int) -> pd.DataFrame:
        res = self.init_res()

        res.indicator = self.df.nvts - self.df.nvts.rolling(window).mean()
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res

    def RUP(self, window:int) -> pd.DataFrame:
        res = self.init_res()

        res.indicator = self.df.unrealized_profit - self.df.unrealized_profit.rolling(window).mean()
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res

    def SSRO(self, window:int) -> pd.DataFrame:
        res = self.init_res()

        res.indicator = self.df.ssr_oscillator - self.df.ssr_oscillator.rolling(window).mean()
        res.position = np.sign(res.indicator)
        res = self.calculate_signal(res)
        return res






    
    
        
        
        
