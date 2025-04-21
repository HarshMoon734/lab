import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, ks_1samp, norm
from statsmodels.graphics.gofplots import qqplot

bse_daily_prices = pd.read_csv("bsedata1.csv", index_col="Date", parse_dates=True)
nse_daily_prices = pd.read_csv("nsedata1.csv", index_col="Date", parse_dates=True)

bse_daily_returns = bse_daily_prices.pct_change().dropna()
nse_daily_returns = nse_daily_prices.pct_change().dropna()

def calculate_descriptive_stats(returns_data, exchange_name):
    stats_summary = pd.DataFrame({
        'Mean': returns_data.mean(),
        'Std Dev': returns_data.std(),
        'Skewness': returns_data.skew(),
        'Kurtosis': returns_data.kurtosis()
    })
    print(f"Descriptive Statistics for {exchange_name}:")
    print(stats_summary, "\n")
    return stats_summary

bse_daily_stats = calculate_descriptive_stats(bse_daily_returns, "BSE Daily")
nse_daily_stats = calculate_descriptive_stats(nse_daily_returns, "NSE Daily")

def plot_returns_boxplot(returns_data, exchange_name):
    plt.figure(figsize=(12, 6))
    plt.boxplot(returns_data)
    plt.title(f"{exchange_name} Returns Boxplot")
    plt.xticks(ticks=range(1, len(returns_data.columns) + 1), 
               labels=returns_data.columns, rotation=20)
    plt.show()

plot_returns_boxplot(bse_daily_returns, "BSE")
plot_returns_boxplot(nse_daily_returns, "NSE")

def plot_qq_charts(returns_data, exchange_name):
    for stock in returns_data.columns[:10]:
        qqplot(returns_data[stock], line='s')
        plt.title(f"Q-Q Plot for {stock} ({exchange_name})")
        plt.show()

plot_qq_charts(bse_daily_returns, "BSE")
plot_qq_charts(nse_daily_returns, "NSE")

def perform_normality_tests(returns_data, exchange_name):
    for stock in returns_data.columns:
        print(f"{stock} ({exchange_name}):")
        
        ks_statistic, ks_pvalue = ks_1samp(returns_data[stock].dropna(), norm.cdf)
        print(f"KS Test: Stat = {ks_statistic:.4f}, p-value = {ks_pvalue:.4f}")
        
        shapiro_stat, shapiro_p = shapiro(returns_data[stock])
        print(f"Shapiro-Wilk: Stat = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}\n")

perform_normality_tests(bse_daily_returns, "BSE")
perform_normality_tests(nse_daily_returns, "NSE")

bse_daily_mle = {stock: stats.norm.fit(bse_daily_returns[stock].dropna()) 
                for stock in bse_daily_returns.columns}
nse_daily_mle = {stock: stats.norm.fit(nse_daily_returns[stock].dropna()) 
                for stock in nse_daily_returns.columns}

def calculate_confidence_intervals(mle_params, returns_data):
    z_score = 1.96
    confidence_intervals = {}

    for stock, (mu, sigma) in mle_params.items():
        n_obs = len(returns_data[stock].dropna())
        std_error = sigma / np.sqrt(n_obs)
        mean_ci = (mu - z_score * std_error, mu + z_score * std_error)

        variance = sigma**2
        var_ci = (
            (n_obs-1) * variance / stats.chi2.ppf(0.975, df=n_obs-1),
            (n_obs-1) * variance / stats.chi2.ppf(0.025, df=n_obs-1)
        )

        confidence_intervals[stock] = {
            "Mean_CI": mean_ci,
            "Variance_CI": var_ci
        }

    return confidence_intervals

bse_daily_ci = calculate_confidence_intervals(bse_daily_mle, bse_daily_returns)
nse_daily_ci = calculate_confidence_intervals(nse_daily_mle, nse_daily_returns)

def print_mle_results(mle_params, ci_data, exchange_name, frequency):
    print(f"\nMLE Estimates for {exchange_name} ({frequency} data):")
    for stock, params in mle_params.items():
        mu, sigma = params
        var = sigma**2
        print(f"{stock}:")
        print(f"  Mean: {mu:.6f} (95% CI: {ci_data[stock]['Mean_CI'][0]:.6f} - {ci_data[stock]['Mean_CI'][1]:.6f})")
        print(f"  Variance: {var:.6f} (95% CI: {ci_data[stock]['Variance_CI'][0]:.6f} - {ci_data[stock]['Variance_CI'][1]:.6f})\n")

print_mle_results(bse_daily_mle, bse_daily_ci, "BSE", "daily")
print_mle_results(nse_daily_mle, nse_daily_ci, "NSE", "daily")

bse_daily_log_returns = np.log(1 + bse_daily_returns)
nse_daily_log_returns = np.log(1 + nse_daily_returns)

bse_log_stats = calculate_descriptive_stats(bse_daily_log_returns, "BSE Daily Log Returns")
nse_log_stats = calculate_descriptive_stats(nse_daily_log_returns, "NSE Daily Log Returns")

bse_weekly_returns = bse_daily_prices.resample('W').ffill().pct_change().dropna()
nse_weekly_returns = nse_daily_prices.resample('W').ffill().pct_change().dropna()
bse_monthly_returns = bse_daily_prices.resample('M').ffill().pct_change().dropna()
nse_monthly_returns = nse_daily_prices.resample('M').ffill().pct_change().dropna()