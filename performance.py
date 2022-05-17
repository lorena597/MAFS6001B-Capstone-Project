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

    # Plot Performance Functions
    def plot_performance(self, plot_column_list):
        # Start Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16,9))
        # Info lines
        for column in plot_column_list:
            ax1.plot(self.df.index, self.df.loc[:, column], label=column)
        ax1.legend(loc='best')
        # Trading Position
        ax2.plot(self.df.index, self.df.loc[:, 'trading_position'], label='Trading position')
        ax2.set_ylabel('Trading Position')
        # Return & Drawdown
        end_mdd = np.argmax(np.maximum.accumulate(self.pnl) - self.pnl) # end of the period
        try:
            start_mdd = np.argmax(self.pnl.iloc[:end_mdd]) # start of period
        except:
            start_mdd = 0
        max_drawdown = max(np.maximum.accumulate(self.pnl) - self.pnl)/np.maximum.accumulate(self.pnl)[end_mdd]
        ax3.plot(df.index, 100*self.pnl, label='Return')
        ax3.plot([end_mdd, start_mdd], [100*self.pnl.iloc[end_mdd], 100*self.pnl.iloc[start_mdd]], 'o', color='Red', markersize=10)
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.legend(loc='best')
        ax3.set_title('Max Drawdown: '+ str(self.df.index.iloc[start_mdd]) + ' ~ ' + str(self.df.index.iloc[end_mdd])+' (-'+str(round((max_drawdown*100),2))+'%)', y=-0.01);
        
    def plot_signals(self, ticker='BTC', n=2000):
        tail = self.df.tail(n).copy()
        plt.figure(figsize=(20, 10))
        plt.plot(tail.loc[:, ticker])
        plt.plot(tail.loc[tail['signal']==1, 'close'], marker='^', markersize=4, color='g', linestyle='None')
        plt.plot(tail.loc[tail['signal']==-1, 'close'], marker='v', markersize=4, color='r', linestyle='None')
        plt.show()
