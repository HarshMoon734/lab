import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math


def black_scholes_pricing(St, t, T, K, r, sigma):
    tau = T-t
    if tau==0:
        return max(0, St-K), max(0,K-St)
    d1 = ((np.log(St/K) + (r+0.5*sigma*sigma) * tau))/(sigma * math.sqrt(tau))
    d2 = d1 - sigma*math.sqrt(tau)

    call_price = St*norm.cdf(d1) - K* np.exp(-r*tau)*norm.cdf(d2)
    put_price = K*np.exp(-r*tau)*norm.cdf(-d2) - St*norm.cdf(-d1)

    return call_price, put_price

def q1():
    call, put = black_scholes_pricing(1, 0, 1, 1, 0.05, 0.6)
    print("Price of Call Option: ", call)
    print("Price of Put Option: ", put)

q1()