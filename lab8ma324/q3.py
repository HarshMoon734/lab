import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

T = 1
K = 1
r = 0.05
sigma = 0.6

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


def plot_3DSurface(St_values, t_values, prices_list, zlabel, str):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(St_values, t_values, prices_list, cmap = 'ocean_r')
    fig.colorbar(surf)

    ax.set_title(str)
    ax.set_xlabel("St Values")
    ax.set_ylabel("t values")
    ax.set_zlabel(zlabel)
    plt.show()


def q3():
    call_list, put_list = [],[]
    St_values = np.linspace(0.01,2,100)
    t_values = np.linspace(0,1,100)

    St_values, t_values = np.meshgrid(St_values, t_values)
    rows = len(St_values)
    cols = len(St_values[0])

    for i in range(rows):
        call_row, put_row = [],[]
        for j in range(cols):
            call,put = black_scholes_pricing(St_values[i][j], t_values[i][i], T,K,r,sigma)
            call_row.append(call)
            put_row.append(put)
        call_list.append(call_row)
        put_list.append(put_row)

    call_list = np.array(call_list)
    put_list = np.array(put_list)


    plot_3DSurface(St_values,t_values,call_list,"C(t,s) surface", "C(t,s) surface plot against s and t")
    plot_3DSurface(St_values,t_values,put_list,"P(t,s) surface", "P(t,s) surface plot against s and t")


q3()