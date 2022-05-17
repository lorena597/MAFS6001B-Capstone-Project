import numpy as np
import cvxpy as cp

'''
Input: 
    mu: (n,1) array
    sigma: (n,n) array
    
Output:
    w: (1,n) array
'''

# # test
# n = 10
# np.random.seed(1)
# mu = np.abs(np.random.randn(n,1))
# sigma = np.random.randn(n,n)
# sigma = sigma.T.dot(sigma)

def EWP(mu, sigma):
    'Equal Weighted Portfolio'
    N = len(mu)
    return np.ones(N)/ N

def MVP(mu, sigma, lmd = 0.5):
    'Mean Variance Portfolio'
    N = len(mu)
    w = cp.Variable(N)
    ret = mu.T @ w
    risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Maximize(ret - lmd * risk), [cp.sum(w) == 1])
    prob.solve()
    return w.value


def GMVP(mu, sigma):
    'Global Minimum Variance Portfolio'
    N = len(mu)
    w = cp.Variable(N)
    risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1])
    prob.solve()
    return w.value

def MSRP(mu, sigma):
    'Maximum Sharpe Ratio Portfolio'
    N = len(mu)
    w_hat = cp.Variable(N)
    risk_hat = cp.quad_form(w_hat, sigma)
    ret_hat = mu.T @ w_hat
    prob = cp.Problem(cp.Minimize(risk_hat), [ret_hat == 1])
    prob.solve()
    return w_hat.value / np.nansum(w_hat.value)

def IVP(mu, sigma):
    'Inverse Volatility Portfolio'
    inverse = 1 / np.sqrt(np.diag(sigma))
    inverse[np.isnan(inverse)] = 0
    return inverse / np.nansum(inverse)

def MDP(mu, sigma):
    'Most Diversified Portfolio'
    return MSRP(np.sqrt(np.diag(sigma)), sigma)

def MDCP(mu, sigma):
    'Maximum Decorrelation Portfolio'
    inverse = np.sqrt(np.diag(sigma)).reshape((-1,1))
    C = sigma / (inverse @ inverse.T)
    return GMVP(mu, C)
