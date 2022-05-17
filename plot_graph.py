import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('total_data.pickle', 'rb') as file:
        total_data = pickle.load(file)
rebalance_times = sorted(list(total_data.keys()))
optimizer = ['MSRP','MDCP']
total_df1 = pd.DataFrame()
total_df2 = pd.DataFrame()
for i in range(7):
    
    with open(f'{i}.pickle', 'rb') as file:
        dict1 = pickle.load(file)
    
    for k in range(len(optimizer)):
        
        portfolio_return = dict1[k]
        if i <= 5:
            time_index = list(pd.date_range(rebalance_times[i], rebalance_times[i+1], freq = 'H'))[1:]
        else:
            time_index = list(pd.date_range(rebalance_times[i], '2022-05-17 14:00:00', freq = 'H'))[1:]
        if len(portfolio_return) > 1440:
            portfolio_return = portfolio_return[1440:]
        
        df = pd.DataFrame({'return': portfolio_return}, index = time_index)

        if k == 0:
            total_df1 = pd.concat([total_df1, df])
        else:
            total_df2 = pd.concat([total_df2, df])
        
total_df1['value'] = (1 + total_df1['return']).cumprod()
total_df2['value'] = (1 + total_df2['return']).cumprod()

total_df1['value'].plot(title = f'Portfolio Value using optimizer {optimizer[0]}')
plt.show()
total_df2['value'].plot(title = f'Portfolio Value using optimizer {optimizer[1]}')
plt.show()

        
        
    
