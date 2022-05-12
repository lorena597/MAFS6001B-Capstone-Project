import numpy as np
import pandas as pd

class Strategy(object):
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def crossover(self, short: int, long: int) -> pd.DataFrame:
        res = pd.DataFrame(index = self.df.index, columns = ["close","signal","position"])
        res.close = self.df.close
        res.signal = self.df.close.rolling(short).mean() - self.df.close.rolling(long).mean()
        res.position = np.sign(res.signal)
        return res
    
    def channelbreakout(self, window: int, threshold: float) -> pd.DataFrame:
        res = pd.DataFrame(index = self.df.index, columns = ["close", "signal", "position"])
        res.close = self.df.close
        res.signal = (self.df.close - self.df.close.rolling(window).mean()) / self.df.close.rolling(window).std()
        res.position = np.sign(res.signal) * (abs(res.signal) > threshold)
        return res
    
    def POS(self, window: int) -> pd.DataFrame:
        
        res = pd.DataFrame(index = self.df.index, columns = ["close", "signal", "position"])
        res.close = self.df.close
        
        H = self.df.high.rolling(window).max().values
        L = self.df.low.rolling(window).min().values
        
        res.signal = (res.close - L) / (H - L)
        res.position = np.sign(res.signal - 0.5)
        return res
            
    def RSI(self, window: int) -> pd.DataFrame:
        res = pd.DataFrame(index = self.df.index, columns = ["close", "signal", "position"])
        res.close = self.df.close
        
        change = self.df.close - self.df.close.shift()
        abs_sum = abs(change).rolling(window).sum()
        change[change < 0] = 0
        up_sum = change.rolling(window).sum()
        res.signal = up_sum / abs_sum * 100
        
        long = (res.signal < 30).values.astype(int)
        short = (res.signal > 70).values.astype(int)
        res.position = long - short
        return res
    
    def OBV(self):
        res = pd.DataFrame(index = self.df.index, columns = ["close", "signal", "position"])
        res.close = self.df.close
        
        sign = np.sign(self.df.close - self.df.close.shift())
        OBV = np.cumsum(self.df.volume * sign)
        
        res.signal = OBV

        return res
    
    def VR(self, window: int) -> pd.DataFrame:
        res = pd.DataFrame(index = self.df.index, columns = ["close","signal","position"])
        res.close = self.df.close
        
        sign = np.sign(self.df.close - self.df.close.shift())
        
        AV = (sign == 1) * self.df.volume
        BV = (sign == -1) * self.df.volume
        CV = (sign == 0) * self.df.volume
        
        AVS = AV.rolling(window).sum()
        BVS = BV.rolling(window).sum()
        CVS = CV.rolling(window).sum()
        
        VR = (AVS + CVS / 2) / (BVS + CVS / 2) * 100
        MAVR = VR.rolling(window).mean()
        
        res.signal = MAVR
        
        long = MAVR < 40
        short = MAVR > 150
        res.position = long.astype(int) - short.astype(int)

        return res
    
    def ADX(self, window: int) -> pd.DataFrame:
        res = pd.DataFrame(index = self.df.index, columns = ["close", "signal", "position"])
        
        DM_plus = self.df.high - self.df.high.shift()
        DM_minus = self.df.low.shift() - self.df.low
        
        DM_plus[DM_plus < 0] = 0
        DM_minus[DM_minus < 0] = 0
        
        mask = DM_plus > DM_minus
        DM_plus = mask * DM_plus
        DM_minus = (~mask) * DM_minus
        
        TR = np.nanmax([self.df.high - self.df.low, \
                        self.df.high - self.df.close.shift(), \
                        self.df.close.shift() - self.df.low], axis = 0)
        TR = pd.Series(TR, index = self.df.index)
        
        DM_plus = self.smooth(DM_plus, window)
        DM_minus = self.smooth(DM_minus, window)
        TR = self.smooth(TR, window)
            
        DI_plus = DM_plus / TR * 100
        DI_minus = DM_minus / TR * 100
        
        DX = abs(DI_plus - DI_minus) / (DI_plus + DI_minus) * 100
        ADX = self.smooth(DX, window) 
        
        res.close = self.df.close
        res.signal = ADX
        
        long = (DI_plus > DI_minus) & (ADX > 50)
        short = (DI_plus < DI_minus) & (ADX > 50)
        res.position = long.astype(int) - short.astype(int)
        
        return res
    
    def smooth(self, series: pd.Series, window: int) -> pd.Series:
        series.fillna(method = "ffill", inplace = True)
        for i in range(len(series)):
            if i == 0 or np.isnan(series[i-1]):
                continue
            else:
                series[i] = (series[i-1] * (window - 1) + series[i]) / window
        return series

    def EFMOM(self, short: int, long: int) -> pd.DataFrame:

        res = pd.DataFrame(index = self.df.index, columns = ["close","signal","position"])
        res.close = self.df.close

        rs = self.df.close / self.df.close.shift(short) - 1
        rl = self.df.close / self.df.close.shift(long) - 1
        pc = self.df.close.pct_change()
        ss = abs(pc).rolling(short).sum()
        sl = abs(pc).rolling(long).sum()
        
        res.signal = rs / ss - rl / sl
        res.position = np.sign(res.signal)
        return res

    def REG(self, window: int) -> pd.DataFrame:

        res = pd.DataFrame(index = self.df.index, columns = ["close","signal","position"])
        res.close = self.df.close

        X = np.ones((window, 2))
        X[:, 1] = np.arange(window) + 1

        def func(y):
            y[np.isnan(y)] = np.nanmean(y)
            return np.linalg.inv(X.T @ X) @ ( X.T @ y)
        
        for i in range(window, len(res)):
            y = res.close[i-window:i]
            b, a = func(y)
            y_hat = y * a + b
            res.signal[i-1] = y[-1] / y_hat[-1] - 1
        
        long = res.signal > 0.05
        short = res.signal < -0.05
        res.position = long.astype(int) - short.astype(int)

        return res

    def SPMOM(self, short: int, long: int) -> pd.DataFrame:

        res = pd.DataFrame(index = self.df.index, columns = ["close", "signal", "position"])
        res.close = self.df.close

        ret = res.close.pct_change()
        sp_short = ret.rolling(short).mean() / ret.rolling(short).std()
        sp_long = ret.rolling(long).mean() / ret.rolling(long).std()

        res.signal = sp_short - sp_long
        res.position = np.sign(res.signal)
        return res
    
    def INVVOL(self, short: int, long: int) -> pd.DataFrame:

        res = pd.DataFrame(index = self.df.index, columns = ["close", "signal", "position"])
        res.close = self.df.close

        ret = res.close.pct_change()
        siv = 1 / ret.rolling(short).std()
        liv = 1 / ret.rolling(long).std()
        res.signal = siv - liv
        res.position = np.sign(res.signal)
        return res

    

        






    
    
        
        
        