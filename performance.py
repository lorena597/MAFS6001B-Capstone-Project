import numpy as np
import pandas as pd

class PerformanceAnalysis(object): 
    
    def __init__(self, pnl: pd.Series, name: str):
        self.pnl = pnl
        self.name = name
        self.ret = self.pnl / self.pnl.shift() - 1

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
        return self.ann_ret / self.ann_vol
        
    @property
    def maxdrawdown(self):
        '''maximum drawdown'''
        mdd = 0
        for i in range(len(self.pnl)):
            mdd = min(mdd, self.pnl[i] / max(self.pnl[:i + 1]) - 1)
        return mdd

    @property
    def turnover(self):
        '''turnover'''
        pass
        return np.nan

    @property
    def calmar(self):
        '''calmar ratio'''
        return self.ann_ret / abs(self.maxdrawdown)
    
    @property
    def VaR(self):
        '''VaR 95%'''
        return np.nanpercentile(self.ret, 5) 
    
    def describe(self):
        dic = {'Annual Ret': self.ann_ret, 'Annual Vol': self.ann_vol, 'Turnover': self.turnover,
                'Max Drawdown': self.maxdrawdown, 'Sharpe Ratio': self.sharpe, 'Calmar Ratio': self.calmar,
                'VaR 95%': self.VaR}
        return pd.DataFrame(dic, index = [self.name])