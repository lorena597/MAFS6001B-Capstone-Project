#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')


# In[18]:


def plot_result(period_num=0):
    BTC_data = pd.read_csv('BTC_price.csv',index_col=0)['close']
    BTC_data.index = pd.to_datetime(BTC_data.index)
    BTC_data = BTC_data/BTC_data.iloc[0]
    with open(str(period_num)+'.pickle', 'rb') as file:
        df = pd.DataFrame(pickle.load(file))
    with open('total_data.pickle', 'rb') as file:
        total_data = pickle.load(file)
    rebalance_times = sorted(list(total_data.keys()))
    optimizer = ['MSRP','MDCP']

    time_index = list(pd.date_range(rebalance_times[0], rebalance_times[1], freq = 'H'))[1:]
    df.index = time_index
    df = (1+df).cumprod()
    compare_df = pd.concat([df, BTC_data], axis=1, join='inner')

    d = {'0':list(compare_df.iloc[:,0]), '1': list(compare_df.iloc[:,1]), 'close': list(compare_df.iloc[:,2])}
    final_df = pd.DataFrame(data=d)
    fig = px.line(final_df, x=final_df.index, y=['0', '1', 'close'])
    newnames = {'0':'MSRP', '1': 'MDCP', 'close': 'HODL BTC'}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                          legendgroup = newnames[t.name],
                                          hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
    fig.show()


# In[19]:


plot_result(0)


# In[ ]:




