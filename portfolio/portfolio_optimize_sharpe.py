import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def fetch_data():
    """Fetches 1-minute data for the specified tickers."""
    tickers = ['AAPL', 'TSLA', 'META', 'AMZN', 'GOOGL']
    print("Fetching 1-minute intraday data for 5 days...")

    # Fetch 2 years of daily data instead of 5 days of 1-minute data
    df = yf.download(tickers, period='2y', interval='1d', progress=False)

    # We only need the 'Close' prices
    df_close = df['Close']

    # Forward-fill any missing data (e.g., if a stock didn't trade for a specific minute)
    # and drop any remaining NaNs
    df_close = df_close.ffill().dropna()
    return df_close

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculates the annualized return and volatility of a portfolio."""
    # There are roughly 252 trading days in a year, and 390 trading minutes in a day.
    # 252 * 390 = 98,280 minutes. We use this to annualize our 1-minute data.

    # But, it is not realistic

    # Change the multiplier from minutes in a year to days in a year
    trading_days_per_year = 252

    annualized_return = np.sum(mean_returns * weights) * trading_days_per_year
    annualized_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days_per_year)

    return annualized_return, annualized_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Returns the negative Sharpe ratio.
    We use negative because scipy's minimize function looks for the lowest number.
    Minimizing the negative Sharpe is the exact same as maximizing the real Sharpe.
    """
    p_ret, p_vol = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_ret - risk_free_rate) / p_vol

def main():
    # 1. Get the data and calculate returns
    df = fetch_data()

    # Calculate logarithmic returns (better for high-frequency financial math)
    returns = np.log(df / df.shift(1)).dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(df.columns)

    # 2. Evaluate the Initial Equal-Weight Portfolio
    initial_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    init_ret, init_vol = calculate_portfolio_performance(initial_weights, mean_returns, cov_matrix)
    init_sharpe = init_ret / init_vol

    print("\n--- Initial Equal-Weight Portfolio ---")
    print(f"Weights: {{ticker: 0.2000 for ticker in df.columns}}")
    print(f"Expected Annual Return: {init_ret * 100:.2f}%")
    print(f"Expected Annual Volatility: {init_vol * 100:.2f}%")
    print(f"Sharpe Ratio: {init_sharpe:.4f}")

    # 3. Optimize for Maximum Sharpe Ratio
    print("\nCalculating optimal weights... (Maximizing Sharpe Ratio)")

    # Constraint: All weights must sum exactly to 1.0 (100% of the portfolio)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

    # Bounds: No short-selling allowed. All weights must be between 0.0 and 1.0
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Run the SciPy optimization engine
    optimized_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP', # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints
    )

    # 4. Extract and Evaluate the Optimal Portfolio
    optimal_weights = optimized_result.x
    opt_ret, opt_vol = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    opt_sharpe = opt_ret / opt_vol

    print("\n--- Optimal Portfolio ---")
    # Clean up the output to 4 decimal places for readability
    clean_weights = {ticker: round(weight, 4) for ticker, weight in zip(df.columns, optimal_weights)}

    for ticker, weight in clean_weights.items():
        print(f"{ticker}: {weight * 100:.2f}%")

    print(f"\nExpected Annual Return: {opt_ret * 100:.2f}%")
    print(f"Expected Annual Volatility: {opt_vol * 100:.2f}%")
    print(f"Sharpe Ratio: {opt_sharpe:.4f}")

if __name__ == "__main__":
    main()
