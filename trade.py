import numpy as np
import pandas as pd

class Trade(object):
    
    def __init__(self, res: pd.DataFrame) -> None:
        self.res = res
        # self.tc = tc * 0.0001 # transaction cost
        # self.sl = 20 # stop loss
    
    def entry_position(self, i: int):
        # 0-->1; 0-->-1; 0-->0;
        self.res.pnl[i] = self.res.pnl[i-1]
        self.res.share[i+1] = self.res.pnl[i] / self.res.close[i] * self.res.position[i]
        
    def hold_position(self, i: int):
        # 0-->0; 1-->1; -1-->-1;
        pct = self.res.close[i] / self.res.close[i-1] - 1.0
        self.res.share[i+1] = self.res.share[i]
        self.res.pnl[i] = self.res.pnl[i-1] * (1 + pct * self.res.position[i])
    
    def exit_position(self, i: int):
        # 1-->0; -1-->0;
        pct = self.res.close[i] / self.res.close[i-1] - 1.0
        self.res.share[i+1] = 0.0
        self.res.pnl[i] = self.res.pnl[i-1] * (1 + pct * self.res.position[i])

    def change_position(self, i: int):
        pct = self.res.close[i] / self.res.close[i-1] - 1.0
        self.res.pnl[i] = self.res.pnl[i-1] * (1 + pct * self.res.position[i])
        self.res.share[i+1] = self.res.pnl[i] / self.res.close[i] * self.res.position[i]
    
    def trade(self):
        
        self.res['pnl'] = 1.0
        self.res["share"] = 0.0
        
        self.res["position"] = self.res["position"].shift()
        self.res["position"][0] = 0
        self.res["position"].fillna(method = "ffill", inplace = True)
               
        for i in range(1, len(self.res)-1):
            
            if self.res.position[i-1] == 0:
                # 0-->1; 0-->-1; 0-->0;
                self.entry_position(i)
                
            elif self.res.position[i-1] == self.res.position[i]:
                # 1-->1; -1-->-1;
                self.hold_position(i)
                
            elif self.res.position[i] == 0:
                # 1-->0; -1-->0;
                self.exit_position(i)
            else:
                # 1-->-1; -1-->1;
                self.change_position(i)
                
        pct = self.res.close[-1] / self.res.close[-2] - 1.0
        self.res.pnl[-1] = self.res.pnl[-2] * (1 + pct * self.res.position[-2])

        return self.res
                
                
            
            
        
            
            
        
        
        
        
        
    
