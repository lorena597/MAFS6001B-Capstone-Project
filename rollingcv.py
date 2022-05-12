import pandas as pd
import numpy as np
from performance import PerformanceAnalysis

from joblib import Parallel, delayed

'''save pnls in a dataframe and set columns as different parameters'''
df = pd.DataFrame(np.random.randn(10000, 5), columns = list(range(5)))

'''cross validation in a rolling window'''

def CV(bgn, end, K):

    whole = df[(bgn <= df.index) & (df.index <= end)]
    whole['ix'] = np.arange(len(whole))
    whole['group'] = pd.cut(whole['ix'], K, labels = list(range(K)))
    test_performance = pd.DataFrame()

    for i in range(K):

        train = whole[whole['group'] != i]; train.reset_index(inplace = True)
        test = whole[whole['group'] == i]; test.reset_index(inplace = True)
        train_performance = pd.DataFrame()

        for j in whole.columns[:-2]:
            
            name = f'leave {i} out {j}'
            train_performance = pd.concat([train_performance, PerformanceAnalysis(train[j], name).describe()])

        rank = train_performance['Sharpe Ratio'].rank()
        col = whole.columns[rank.argmax()]
        test_performance = pd.concat([test_performance, PerformanceAnalysis(test[col], f'leave {i} out {col}').describe()])

    rank = test_performance['Sharpe Ratio'].rank()
    best_parameter = whole.columns[rank.argmax()]
    return best_parameter

'''rolling cross validation'''
N = len(df)
window_size = 6 * 24
step = 2 * 24
bgn_ix = np.array(range(0, N - window_size, step))
end_ix = bgn_ix + window_size
params = pd.Series(np.zeros(N)); params[:] = np.nan

def func(i):
    bgn_i = df.index[bgn_ix[i]]
    end_i = df.index[end_ix[i]]
    return end_ix[i], CV(bgn_i, end_i, 6)

res = Parallel(n_jobs = 10)(delayed(func)(i) for i in range(len(bgn_ix)))

for r in res:
    i, p = r
    params[i] = p

params.fillna(method = 'ffill', inplace = True)






