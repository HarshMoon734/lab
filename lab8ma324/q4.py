import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tabulate import tabulate

St_values = [0.4, 0.6, 0.8, 1.0, 1.2]
t = 0
T = 1
K = 1
r = 0.05
sigma = 0.6


def black_scholes_pricing(St, t, T, K, r, sigma):
    tau = T - t
    if tau == 0:
        return max(0, St-K), max(0, K-St)
    d1 = ((np.log(St / K) + (r + 0.5 * sigma * sigma) * tau)) / \
        (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    call_price = St * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    put_price = K * np.exp(-r * tau) * norm.cdf(-d2) - St * norm.cdf(-d1)

    return call_price, put_price


def sensitivity_T(T_sample, K, r, sigma):
    T_values = np.linspace(0.1, 5, num=500)
    call_list, put_list = [], []
    counter = 0
    data = []
    for St in St_values:
        call, put = [], []
        for T in T_values:
            C, P = black_scholes_pricing(St, t, T, K, r, sigma)
            call.append(C)
            put.append(P)

            if St == 0.8:
                if counter % 50 == 0:
                    data.append([1 + int(counter/50), T, C, P])
                counter += 1

        call_list.append(call)
        put_list.append(put)

    print("------------------------- Sensitivity Analysis with respect to T -------------------------\n")
    heading = ['S. No', 'T', 'C(t,St)', 'P(t,St)']
    print(tabulate(data, headers=heading))

    for idx in range(len(St_values)):
        plt.plot(T_values, call_list[idx],
                 label='x = {}'.format(St_values[idx]))
    plt.xlabel('T')
    plt.ylabel('C(t,St)')
    plt.title('Plot for C(t,St) vs T')
    plt.legend()
    plt.grid()
    plt.show()

    for idx in range(len(St_values)):
        plt.plot(T_values, put_list[idx],
                 label='St = {}'.format(St_values[idx]))

    plt.xlabel('T')
    plt.ylabel('P(t,St)')
    plt.title('Plot for P(t,St) vs T')
    plt.legend()
    plt.grid()
    plt.show()


def sensitivity_K(T, K_sample, r, sigma):
    K_list = np.linspace(0.1, 2, num=500)
    call_list, put_list = [], []

    counter = 0
    data = []

    for x in St_values:
        call, put = [], []
        for K in K_list:
            C, P = black_scholes_pricing(x, t, T, K, r, sigma)
            call.append(C)
            put.append(P)

            if x == 0.8:
                if counter % 50 == 0:
                    data.append([1 + int(counter/50), K, C, P])
                counter += 1

        call_list.append(call)
        put_list.append(put)

    print("------------------------- Sensitivity Analysis with respect to K -------------------------\n")
    heading = ['S. No', 'K', 'C(t,St)', 'P(t,St)']
    print(tabulate(data, headers=heading))

    for idx in range(len(St_values)):
        plt.plot(K_list, call_list[idx], label='St = {}'.format(St_values[idx]))

    plt.xlabel('K')
    plt.ylabel('C(t,St)')
    plt.title('Plot for C(t,St) vs K')
    plt.legend()
    plt.grid()
    plt.show()

    for idx in range(len(St_values)):
        plt.plot(K_list, put_list[idx], label='St = {}'.format(St_values[idx]))

    plt.xlabel('K')
    plt.ylabel('P(t,St)')
    plt.title('Plot for P(t,St) vs K')
    plt.legend()
    plt.grid()
    plt.show()


def sensitivity_r(T, K, r_sample, sigma):

    r_list = np.linspace(0, 1, num=500, endpoint=False)
    call_list, put_list = [], []

    counter = 0
    data = []

    for x in St_values:
        call, put = [], []
        for r in r_list:
            C, P = black_scholes_pricing(x, t, T, K, r, sigma)
            call.append(C)
            put.append(P)

            if x == 0.8:
                if counter % 50 == 0:
                    data.append([1 + int(counter/50), r, C, P])
                counter += 1

        call_list.append(call)
        put_list.append(put)

    print("------------------------- Sensitivity Analysis with respect to r -------------------------\n")
    heading = ['S. No', 'r', 'C(t,St)', 'P(t,St)']
    print(tabulate(data, headers=heading))

    for idx in range(len(St_values)):
        plt.plot(r_list, call_list[idx], label='St = {}'.format(St_values[idx]))

    plt.xlabel('r')
    plt.ylabel('C(t,St)')
    plt.title('Plot for C(t,St) vs r')
    plt.legend()
    plt.grid()
    plt.show()

    for idx in range(len(St_values)):
        plt.plot(r_list, put_list[idx], label='St = {}'.format(St_values[idx]))

    plt.xlabel('r')
    plt.ylabel('P(t,St)')
    plt.title('Plot for P(t,St) vs r')
    plt.legend()
    plt.grid()
    plt.show()


def sensitivity_sigma(T, K, r, sigma_sample):

    sigma_list = np.linspace(0.001, 1, num=500, endpoint=False)
    call_list, put_list = [], []

    counter = 0
    data = []

    for x in St_values:
        call, put = [], []
        for sigma in sigma_list:
            C, P = black_scholes_pricing(x, t, T, K, r, sigma)
            call.append(C)
            put.append(P)

            if x == 0.8:
                if counter % 50 == 0:
                    data.append([1 + int(counter/50), sigma, C, P])
                counter += 1

        call_list.append(call)
        put_list.append(put)

    heading = ['S. No', 'sigma', 'C(t,St)', 'P(t,St)']
    print(tabulate(data, headers=heading))

    for idx in range(len(St_values)):
        plt.plot(sigma_list, call_list[idx],
                 label='St = {}'.format(St_values[idx]))

    plt.xlabel('sigma')
    plt.ylabel('C(t,St)')
    plt.title('Plot for C(t,St) vs sigma')
    plt.legend()
    plt.grid()
    plt.show()

    for idx in range(len(St_values)):
        plt.plot(sigma_list, put_list[idx],
                 label='St = {}'.format(St_values[idx]))

    plt.xlabel('sigma')
    plt.ylabel('P(t,St)')
    plt.title('Plot for P(t,St) vs sigma')
    plt.legend()
    plt.grid()
    plt.show()


def sensitivity_K_and_r(x, t, T, sigma):
    call_list, put_list = [], []
    K_list = np.linspace(0.01, 2, num=100)
    r_list = np.linspace(0, 1, num=100, endpoint=False)

    K_list, r_list = np.meshgrid(K_list, r_list)
    row, col = len(K_list), len(K_list[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                x, t, T, K_list[i][j], r_list[i][j], sigma)
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, r_list, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs K and r')
    ax.set_xlabel("K")
    ax.set_ylabel("r")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, r_list, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs K and r')
    ax.set_xlabel("K")
    ax.set_ylabel("r")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_K_and_sigma(x, t, T, r):
    call_list, put_list = [], []
    K_list = np.linspace(0.01, 2, num=100)
    sigma_list = np.linspace(0.01, 1, num=100, endpoint=False)

    K_list, sigma_list = np.meshgrid(K_list, sigma_list)
    row, col = len(K_list), len(K_list[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                x, t, T, K_list[i][j], r, sigma_list[i][j])
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, sigma_list, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs K and sigma')
    ax.set_xlabel("K")
    ax.set_ylabel("sigma")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, sigma_list, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs K and sigma')
    ax.set_xlabel("K")
    ax.set_ylabel("sigma")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_r_and_sigma(x, t, T, K):
    call_list, put_list = [], []
    sigma_list = np.linspace(0.01, 1, num=100, endpoint=False)
    r_list = np.linspace(0.001, 1, num=100, endpoint=False)

    sigma_list, r_list = np.meshgrid(sigma_list, r_list)
    row, col = len(sigma_list), len(sigma_list[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                x, t, T, K, r_list[i][j], sigma_list[i][j])
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(sigma_list, r_list, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs sigma and r')
    ax.set_xlabel("sigma")
    ax.set_ylabel("r")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(sigma_list, r_list, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs sigma and r')
    ax.set_xlabel("sigma")
    ax.set_ylabel("r")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_T_and_K(x, t, r, sigma):
    call_list, put_list = [], []
    K_list = np.linspace(0.01, 2, num=100)
    T_values = np.linspace(0.1, 5, num=100)

    K_list, T_values = np.meshgrid(K_list, T_values)
    row, col = len(K_list), len(K_list[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                x, t, T_values[i][j], K_list[i][j], r, sigma)
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, T_values, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs K and T')
    ax.set_xlabel("K")
    ax.set_ylabel("T")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, T_values, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs K and T')
    ax.set_xlabel("K")
    ax.set_ylabel("T")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_T_and_r(x, t, K, sigma):
    call_list, put_list = [], []
    r_list = np.linspace(0.01, 1, num=100)
    T_values = np.linspace(0.1, 5, num=100)

    r_list, T_values = np.meshgrid(r_list, T_values)
    row, col = len(r_list), len(r_list[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                x, t, T_values[i][j], K, r_list[i][j], sigma)
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T_values, r_list, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs T and r')
    ax.set_xlabel("T")
    ax.set_ylabel("r")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T_values, r_list, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs T and r')
    ax.set_xlabel("T")
    ax.set_ylabel("r")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_T_and_sigma(x, t, K, r):
    call_list, put_list = [], []
    sigma_list = np.linspace(0.01, 1, num=100)
    T_values = np.linspace(0.1, 5, num=100)

    sigma_list, T_values = np.meshgrid(sigma_list, T_values)
    row, col = len(sigma_list), len(sigma_list[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                x, t, T_values[i][j], K, r, sigma_list[i][j])
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T_values, sigma_list, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs T and sigma')
    ax.set_xlabel("T")
    ax.set_ylabel("sigma")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T_values, sigma_list, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs T and sigma')
    ax.set_xlabel("T")
    ax.set_ylabel("sigma")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_K_and_x(t, T, r, sigma):
    call_list, put_list = [], []
    K_list = np.linspace(0.01, 2, num=100)
    St_values = np.linspace(0.2, 2, num=100)

    K_list, St_values = np.meshgrid(K_list, St_values)
    row, col = len(St_values), len(St_values[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                St_values[i][j], t, T, K_list[i][j], r, sigma)
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, St_values, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs K and St')
    ax.set_xlabel("K")
    ax.set_ylabel("St")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(K_list, St_values, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs K and St')
    ax.set_xlabel("K")
    ax.set_ylabel("St")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_T_and_x(t, K, r, sigma):
    call_list, put_list = [], []
    St_values = np.linspace(0.2, 2, num=100)
    T_values = np.linspace(0.1, 5, num=100)

    St_values, T_values = np.meshgrid(St_values, T_values)
    row, col = len(St_values), len(St_values[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                St_values[i][j], t, T_values[i][j], K, r, sigma)
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T_values, St_values, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs T and St')
    ax.set_xlabel("T")
    ax.set_ylabel("St")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T_values, St_values, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs T and St')
    ax.set_xlabel("T")
    ax.set_ylabel("St")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_x_and_r(t, K, T, sigma):
    call_list, put_list = [], []
    St_values = np.linspace(0.2, 2, num=100)
    r_list = np.linspace(0.01, 1, num=100)

    St_values, r_list = np.meshgrid(St_values, r_list)
    row, col = len(St_values), len(St_values[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                St_values[i][j], t, T, K, r_list[i][j], sigma)
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(St_values, r_list, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs St and r')
    ax.set_xlabel("St")
    ax.set_ylabel("r")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(St_values, r_list, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs St and r')
    ax.set_xlabel("St")
    ax.set_ylabel("r")
    ax.set_zlabel("P(t,St)")
    plt.show()


def sensitivity_x_and_sigma(t, K, T, r):
    call_list, put_list = [], []
    St_values = np.linspace(0.2, 2, num=100)
    sigma_list = np.linspace(0.01, 1, num=100)

    St_values, sigma_list = np.meshgrid(St_values, sigma_list)
    row, col = len(St_values), len(St_values[0])

    for i in range(row):
        call_list.append([])
        put_list.append([])

        for j in range(col):
            C, P = black_scholes_pricing(
                St_values[i][j], t, T, K, r, sigma_list[i][j])
            call_list[i].append(C)
            put_list[i].append(P)

    call_list = np.array(call_list)
    put_list = np.array(put_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(St_values, sigma_list, call_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('C(t,St) vs St and sigma')
    ax.set_xlabel("St")
    ax.set_ylabel("sigma")
    ax.set_zlabel("C(t,St)")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(St_values, sigma_list, put_list, cmap='ocean_r')
    fig.colorbar(surf)
    plt.title('P(t,St) vs St and sigma')
    ax.set_xlabel("St")
    ax.set_ylabel("sigma")
    ax.set_zlabel("P(t,St)")
    plt.show()


def q4():
    sensitivity_T(1, 1, 0.05, 0.6)
    sensitivity_K(1, 1, 0.05, 0.6)
    sensitivity_r(1, 1, 0.05, 0.6)
    sensitivity_sigma(1, 1, 0.05, 0.6)
    sensitivity_K_and_r(0.8, 0, 1, 0.6)
    sensitivity_K_and_sigma(0.8, 0, 1, 0.05)
    sensitivity_r_and_sigma(0.8, 0, 1, 1)
    sensitivity_T_and_K(0.8, 0, 0.05, 0.6)
    sensitivity_T_and_r(0.8, 0, 1, 0.6)
    sensitivity_T_and_sigma(0.8, 0, 1, 0.05)
    sensitivity_K_and_x(0, 1, 0.05, 0.6)
    sensitivity_T_and_x(0, 1, 0.05, 0.6)
    sensitivity_x_and_r(0, 1, 1, 0.6)
    sensitivity_x_and_sigma(0, 1, 1, 0.05)



q4()
