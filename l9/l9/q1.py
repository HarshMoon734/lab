import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

r = 0.05
t = 0
T = 6/12
bse_stocks = ['RELIANCE_BO', 'TCS_BO', 'HDFCBANK_BO', 'HINDUNILVR_BO', 'INFY_BO', 'KOTAKBANK_BO', 'ICICIBANK_BO', 'LT_BO', 'HDFC_BO', 'SBIN_BO']
other_bse_stocks = ['GOOGL', 'AAPL', 'AMZN', 'MSFT', 'NVDA', 'GOOG', 'NFLX', 'TSLA', 'INTC', 'CSCO']

nse_stocks = ['RELIANCE_NS', 'TCS_NS', 'HINDUNILVR_NS', 'INFY_NS', 'KOTAKBANK_NS', 'ICICIBANK_NS', 'LT_NS', 'SBIN_NS', 'ITC_NS', 'ONGC_NS']
other_nse_stocks = ['BABA', 'BIDU', 'NVDA', 'JD', 'PYPL', 'SNAP', 'MCD', 'UBER', 'LYFT', 'SQ']

def get_historical_volatility(stocks_type, time_period):
  filename, stocks_name = '', []
  if stocks_type == 'BSE':
    stocks_name = bse_stocks + other_bse_stocks
    filename = './bsedata1.csv'
  else:
    stocks_name = nse_stocks + other_nse_stocks
    filename = './nsedata1.csv'
  
  df = pd.read_csv(filename)
  # df=df.pct_change()
  df_monthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)

  start_idx = 60 - time_period
  df_reduced = df_monthly.iloc[start_idx :]
  df_reduced.reset_index(inplace = True, drop = True) 
  idx_list = df.index[df['Date'] >= df_reduced.iloc[0]['Date']].tolist()
  df_reduced = df.iloc[idx_list[0] :]

  data = df_reduced.set_index('Date')
  data = data.pct_change()

  volatility = []
  for sname in stocks_name:
    returns = data[sname]
    x = returns.to_list()
    mean = np.nanmean(np.array(x))
    std = np.nanstd(np.array(x))
    
    volatility.append(std * math.sqrt(252))
  
  table = []
  for i in range(len(volatility)):
    table.append([i + 1, stocks_name[i], volatility[i]])
  
  return volatility


def BSM_model(x, t, T, K, r, sigma):
  if t == T:
    return max(0, x - K), max(0, K - x)
  d1 = ( math.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  d2 = ( math.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  call_price = x * norm.cdf(d1) - K * math.exp( -r * (T - t) ) * norm.cdf(d2)
  put_price = K * math.exp( -r * (T - t) ) * norm.cdf(-d2) - x * norm.cdf(-d1)
  return call_price, put_price

print("########################## Part a #####################################")
print("Calculating Historical volatility for BSE....")
stocks_name1 = bse_stocks + other_bse_stocks
df1 = pd.read_csv('./bsedata1.csv')
df_monthly = df1.groupby(pd.DatetimeIndex(df1.Date).to_period('M')).nth(0)
df_reduced = df_monthly.iloc[59 :]
df_reduced.reset_index(inplace = True, drop = True) 
idx_list = df1.index[df1['Date'] >= df_reduced.iloc[0]['Date']].tolist()
df_reduced = df1.iloc[idx_list[0] :]
data = df_reduced.set_index('Date')
data = data.pct_change()
volatility1 = []
df1 = df1.interpolate(method ='linear', limit_direction ='forward')
for sname in stocks_name1:
  returns = data[sname]
  x = returns.to_list()
  mean = np.nanmean(np.array(x))
  std = np.nanstd(np.array(x))
  volatility1.append(std * math.sqrt(252))
table = []
for i in range(len(volatility1)):
  print("------------------------------------------")
  print(f"{i+1}. stock name: {stocks_name1[i]}")
  print(f"volatility:{volatility1[i]:.6f}")
  print("------------------------------------------\n")

print("\n")
print("\n\nCalculating Historical volatility for NSE....")
stocks_name2 = nse_stocks + other_nse_stocks
df2 = pd.read_csv('./nsedata1.csv')
df_monthly = df2.groupby(pd.DatetimeIndex(df2.Date).to_period('M')).nth(0)
df_reduced = df_monthly.iloc[59 :]
df_reduced.reset_index(inplace = True, drop = True) 
idx_list = df2.index[df2['Date'] >= df_reduced.iloc[0]['Date']].tolist()
df_reduced = df2.iloc[idx_list[0] :]
data = df_reduced.set_index('Date')
data = data.pct_change()
volatility2 = []
df2 = df2.interpolate(method ='linear', limit_direction ='forward')
for sname in stocks_name2:
  returns = data[sname]
  x = returns.to_list()
  mean = np.nanmean(np.array(x))
  std = np.nanstd(np.array(x))
  
  volatility2.append(std * math.sqrt(252))

table = []
for i in range(len(volatility2)):
  print("------------------------------------------")
  print(f"{i+1}. stock name: {stocks_name2[i]}")
  print(f"volatility:{volatility2[i]:.6f}")
  print("------------------------------------------\n")

print("\n\n################################## part b ##############################\n")
for iterator in range(len(stocks_name1)):
    print(f"\n\nStock name - {stocks_name1[iterator]}")
    sigma = volatility1[iterator]
    print("Historical volatility for last 1 month\t=", sigma, "\n")
    S0 = df1.iloc[len(df1) - 1][stocks_name1[iterator]]
    table = []

    for idx2 in range(5, 16):
      K = S0 * round(idx2 * 0.1, 2)
      call, put = BSM_model(S0, 0, T, K, r, sigma)
      table.append([str(round(idx2 * 0.1, 2)) + "*S0", call, put])
      print(f"K={K:.6f}   ,  call={call:.6f}  ,  put={put:.6f}")
      
for iterator in range(len(stocks_name2)):
    print(f"\n\nStock Name: {stocks_name2[iterator]} :-")
    sigma = volatility2[iterator]
    print("Historical volatility for last 1 month\t=", sigma, "\n")
    S0 = df2.iloc[len(df2) - 1][stocks_name2[iterator]]
    table = []

    for idx2 in range(5, 16):
      K = S0 * round(idx2 * 0.1, 2)
      call, put = BSM_model(S0, 0, T, K, r, sigma)
      print(f"K={K:.6f}   ,  call={call:.6f}  ,  put={put:.6f}")

print("\n\n########################### part c #########################\n")
sigma_list=[]
time_period = range(1, 61)
for delta_t in range(1, 61):
    sigma_list.append(get_historical_volatility('BSE', delta_t))
    
for iterator in range(len(stocks_name1)):
  print(f"Stock name : {stocks_name1[iterator]} :-")
  plt.rcParams["figure.figsize"] = (20, 12)
  S0 = df1.iloc[len(df1) - 1][stocks_name1[iterator]]
  call_prices, put_prices = np.zeros((21, 60)), np.zeros((21, 60))
  historical_volatility = []
  
  for idx2 in range(60):
    sigma = sigma_list[idx2][iterator]
    historical_volatility.append(sigma)
    A = [round(0.1 * i, 2) for i in range(5, 16)]

    for idx3 in range(len(A)):
      K = A[idx3] * S0
      call, put = BSM_model(S0, t, T, K, r, sigma)
      call_prices[idx3][idx2] = call
      put_prices[idx3][idx2] = put
    
  for i in range(len(A)):
    ax = plt.subplot(2, 2, (i % 4) + 1)
    plt.plot(time_period, call_prices[i])
    plt.xlabel("time period (months)")
    plt.ylabel("European Call Option Price")
    ax.set_title(f"Call Option for {stocks_name1[iterator]} with Strike price K = {A[i]}*S0")
    if (i+1) == 4:
      plt.savefig('./call_BSE/' + stocks_name1[iterator] + '_Call_prices_set1.jpg')
      plt.close() 
    if (i+1) == 8:
      plt.savefig('./call_BSE/' + stocks_name1[iterator] + '_Call_prices_set2.jpg')
      plt.close()
  plt.savefig('./call_BSE/' + stocks_name1[iterator] + '_Call_prices_set3.jpg')
  plt.close()

  for i in range(len(A)):
    ax = plt.subplot(2, 2, (i % 4) + 1)
    plt.plot(time_period, put_prices[i])
    plt.xlabel("time period (months)")
    plt.ylabel("European Put Option Price")
    ax.set_title("Put Option for {} with Strike price K = {}*S0".format(stocks_name1[iterator], A[i]))
    if i == 3:
      plt.savefig('./put_BSE/' + stocks_name1[iterator] + '_Put_prices_set1.jpg')
      # print(i)
      # plt.legend()
      plt.close()
    if i == 7:
      plt.savefig('./put_BSE/' + stocks_name1[iterator] + '_Put_prices_set2.jpg')
      plt.close()
  plt.savefig('./put_BSE/' + stocks_name1[iterator] + '_Put_prices_set3.jpg')
  plt.close()

  plt.plot(time_period, historical_volatility)
  plt.xlabel("time period (months)")
  plt.ylabel("Volatility")
  title=plt.title("{}".format(stocks_name1[iterator]))
  title.set_fontsize(26)
  title.set_fontweight("bold")
  plt.savefig('./volBSE/' + stocks_name1[iterator] + '_Historical_volatility.jpg')
  plt.close()

sigma_list=[]
time_period = range(1, 61)
for delta_t in range(1, 61):
    sigma_list.append(get_historical_volatility('NSE', delta_t))
    
for iterator in range(len(stocks_name2)):
  print(f"Stock name : {stocks_name2[iterator]} :-")
  plt.rcParams["figure.figsize"] = (20, 12)
  S0 = df2.iloc[len(df2) - 1][stocks_name2[iterator]]
  call_prices, put_prices = np.zeros((21, 60)), np.zeros((21, 60))
  historical_volatility = []
  
  for idx2 in range(60):
    sigma = sigma_list[idx2][iterator]
    historical_volatility.append(sigma)
    A = [round(0.1 * i, 2) for i in range(5, 16)]

    for idx3 in range(len(A)):
      K = A[idx3] * S0
      call, put = BSM_model(S0, t, T, K, r, sigma)
      call_prices[idx3][idx2] = call
      put_prices[idx3][idx2] = put
    
  for i in range(len(A)):
    ax = plt.subplot(2, 2, (i % 4) + 1)
    plt.plot(time_period, call_prices[i])
    plt.xlabel("time period (months)")
    plt.ylabel("European Call Option Price")
    suptitle=plt.suptitle(f"{stocks_name2[iterator]}")
    suptitle.set_fontsize(20)
    suptitle.set_fontweight('bold')
    ax.set_title("Call Option for {} with Strike price K = {}*S0".format(stocks_name2[iterator], A[i]))
    if i == 3:
      plt.savefig('./call_NSE/' + stocks_name2[iterator] + '_Call_prices_set1.jpg')
      plt.close() 
    if i == 7:
      plt.savefig('./call_NSE/' + stocks_name2[iterator] + '_Call_prices_set2.jpg')
      plt.close()
  plt.savefig('./call_NSE/' + stocks_name2[iterator] + '_Call_prices_set3.jpg')
  plt.close()

  for i in range(len(A)):
    ax = plt.subplot(2, 2, (i % 4) + 1)
    plt.plot(time_period, put_prices[i])
    plt.xlabel("time period (months)")
    plt.ylabel("European Put Option Price")
    suptitle=plt.suptitle(f"{stocks_name2[iterator]}")
    suptitle.set_fontsize(20)
    suptitle.set_fontweight('bold')
    ax.set_title("Put Option for {} with Strike price K = {}*S0".format(stocks_name2[iterator], A[i]))
    if i == 3:
      plt.savefig('./put_NSE/' + stocks_name2[iterator] + '_Put_prices_set1.jpg')
      plt.close()
    if i == 7:
      plt.savefig('./put_NSE/' + stocks_name2[iterator] + '_Put_prices_set2.jpg')
      plt.close()
  plt.savefig('./put_NSE/' + stocks_name2[iterator] + '_Put_prices_set3.jpg')
  plt.close()

  plt.plot(time_period, historical_volatility)
  plt.xlabel("time period (months)")
  plt.ylabel("Volatility")
  title=plt.title("{}".format(stocks_name2[iterator]))
  title.set_fontsize(26)
  title.set_fontweight("bold")
  plt.savefig('./volNSE/' + stocks_name2[iterator] + '_Historical_volatility.jpg')
  plt.close()