import numpy as np
import pandas as pd

class PerformanceAnalysis(object): 
    
    def __init__(self, df: pd.DataFrame, name: str, winrate: float):
        
        self.df = df
        self.name = name
        self.pnl = self.df.portfolio_value
        self.ret = self.df.portfolio_return
        self.winrate = winrate

    @property
    def ann_ret(self):
        '''annual return'''
        return np.nanmean(self.ret) * 24 * 365
    
    @property
    def ann_vol(self):
        '''annual volatility'''
        return np.nanstd(self.ret) * np.sqrt(365 * 24)
    
    @property
    def sharpe(self):
        '''sharpe ratio'''
        if self.ann_vol != 0:
            sharpe = self.ann_ret / self.ann_vol
        else:
            sharpe = np.nan
        return sharpe
        
    @property
    def maxdrawdown(self):
        '''maximum drawdown'''
        mdd = 0
        for i in range(len(self.pnl)):
            mdd = min(mdd, self.pnl[i] / max(self.pnl[:i + 1]) - 1)
        return mdd

    @property
    def calmar(self):
        '''calmar ratio'''
        if self.maxdrawdown != 0:
            calma = self.ann_ret / abs(self.maxdrawdown)
        else:
            calma = np.nan
        return calma
    
    @property
    def VaR(self):
        '''VaR 95%'''
        return np.nanpercentile(self.ret, 5) 
    
    def describe(self):
        dic = {'Annual Ret': self.ann_ret, 'Annual Vol': self.ann_vol, 'Max Drawdown': self.maxdrawdown, 
                'Sharpe Ratio': self.sharpe, 'Calmar Ratio': self.calmar, 'VaR 95%': self.VaR, 'Winrate': self.winrate}
        return pd.DataFrame(dic, index = [self.name])
