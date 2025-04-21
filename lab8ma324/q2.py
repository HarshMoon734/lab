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
    # Using put-call parity C + K = S + P
    put_price = K*np.exp(-r*tau)*norm.cdf(-d2) - St*norm.cdf(-d1)

    return call_price, put_price


def plot_2DGraph(St_values, t_values, prices_list, ylabel, str):
    for t in range(len(t_values)):
        call_prices = prices_list[t]
        plt.plot(St_values,call_prices,label = f't = {t}')
    plt.xlabel('St values')
    plt.ylabel(ylabel)
    plt.title(str)
    plt.legend()
    plt.grid()
    plt.show()

def plot_3DGraph(St_values, t_values, prices_list, zlabel, str):
    x,y,z = [],[],[]
    for i in range(len(t_values)):
        for j in range(len(St_values)):
            x.append(St_values[j])
            y.append(t_values[i])
            z.append(prices_list[i][j])
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(x,y,z)
    ax.set_xlabel('St_values')
    ax.set_ylabel('t_values')
    ax.set_zlabel(zlabel)
    plt.title(str)
    plt.show()


def q2():
    T = 1
    K = 1
    r = 0.05
    sigma = 0.6
    t_values = [0,0.2,0.4,0.6,0.8,1]
    St_values = np.linspace(0.1,2,1000)


    call_list, put_list = [],[]
    for t in t_values:
        call_prices,put_prices = [],[]
        for St in St_values:
            call,put = black_scholes_pricing(St,t,T,K,r,sigma)
            call_prices.append(call)
            put_prices.append(put)
        
        call_list.append(call_prices)
        put_list.append(put_prices)

    plot_2DGraph(St_values,t_values,call_list,"C(t,s) v/s St", "2D-plot of C(t,s) v/s s")
    plot_2DGraph(St_values,t_values,put_list,"P(t,s) v/s St", "2D-plot of P(t,s) v/s s")
    plot_3DGraph(St_values,t_values,call_list, "C(t,s) v/s St and t", "3D-plot of C(t,s) v/s t and s")
    plot_3DGraph(St_values,t_values,put_list, "P(t,s) v/s St and t", "3D-plot of P(t,s) v/s t and s")

q2()