import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

def fetch_data(ticker, period="5y"):
    """Fetches historical data and handles yfinance formatting updates."""
    print(f"Fetching historical data for {ticker}...")
    df = yf.download(ticker, period=period)

    # Safeguard for recent yfinance updates (flattens MultiIndex columns if present)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[['Close']].copy()

def backtest_sma(df_input, short_window, long_window):
    """Calculates signals and returns for a given pair of moving averages."""
    df = df_input.copy()

    # 1. Calculate Moving Averages
    df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()

    # Drop rows where we don't have enough data to calculate the long SMA
    df.dropna(inplace=True)

    # 2. Generate Signals (1 = Buy/Hold, 0 = Sell/Flat)
    # We buy when the Short SMA is strictly greater than the Long SMA
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)

    # 3. Calculate Daily Returns
    df['Market_Return'] = df['Close'].pct_change()

    # 4. Calculate Strategy Returns
    # CRITICAL: We must shift the signal by 1 day!
    # If the MAs cross today, we can't buy at today's closing price. We buy tomorrow.
    # Without this shift, your backtest will suffer from "Look-Ahead Bias" and be falsely profitable.
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Market_Return']

    # Calculate Cumulative Returns to find the total profit multiplier
    df['Cumulative_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()

    # Get the final return value (e.g., 2.5 means a 150% profit)
    total_profit = df['Cumulative_Strategy_Return'].iloc[-1] if not df.empty else 0

    return total_profit, df

def main():
    ticker = "AAPL"
    df = fetch_data(ticker, period="5y")

    # Define the ranges for our Grid Search
    short_windows = range(10, 60, 5)   # Tests 10, 15, 20... up to 55
    long_windows = range(50, 210, 10)  # Tests 50, 60, 70... up to 200

    best_profit = 0
    best_params = (0, 0)
    best_df = None

    print("Starting grid search for the most profitable SMA parameters...")

    # Execute the Grid Search
    for sw, lw in itertools.product(short_windows, long_windows):
        if sw >= lw:
            continue # The short window must be smaller than the long window

        profit, result_df = backtest_sma(df, sw, lw)

        # If this combination is better than our previous best, save it
        if profit > best_profit:
            best_profit = profit
            best_params = (sw, lw)
            best_df = result_df

    print("\n--- Grid Search Complete ---")
    print(f"Best Short Window: {best_params[0]} days")
    print(f"Best Long Window:  {best_params[1]} days")
    print(f"Strategy Return:   {(best_profit - 1) * 100:.2f}%")

    # --- Visualization ---
    print("\nGenerating performance plots...")
    plt.figure(figsize=(14, 10))

    # Top Chart: Price, Moving Averages, and Buy/Sell Markers
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(best_df.index, best_df['Close'], label='AAPL Close', color='gray', alpha=0.5)
    ax1.plot(best_df.index, best_df['SMA_Short'], label=f'{best_params[0]}-Day SMA', color='dodgerblue')
    ax1.plot(best_df.index, best_df['SMA_Long'], label=f'{best_params[1]}-Day SMA', color='darkorange')

    # Calculate exact crossover points to plot markers
    best_df['Position_Change'] = best_df['Signal'].diff()
    buy_signals = best_df[best_df['Position_Change'] == 1]
    sell_signals = best_df[best_df['Position_Change'] == -1]

    ax1.scatter(buy_signals.index, buy_signals['SMA_Short'], marker='^', color='green', s=120, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['SMA_Short'], marker='v', color='red', s=120, label='Sell Signal', zorder=5)

    ax1.set_title(f'AAPL - Optimized SMA Crossover ({best_params[0]} & {best_params[1]})', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Bottom Chart: Cumulative Equity Curve
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(best_df.index, best_df['Cumulative_Market_Return'], label='Buy & Hold Equity', color='gray', alpha=0.7)
    ax2.plot(best_df.index, best_df['Cumulative_Strategy_Return'], label='Strategy Equity', color='purple', linewidth=2)

    ax2.set_title('Equity Curve: Strategy vs. Buy & Hold', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Return Multiplier (1.0 = Breakeven)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
