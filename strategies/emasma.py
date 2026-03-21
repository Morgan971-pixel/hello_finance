import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Define the stock ticker and the time period
    ticker_symbol = 'AAPL' # You can change this to any ticker, e.g., 'MSFT', 'TSLA'
    print(f"Fetching historical data for {ticker_symbol}...")

    # Download 1 year of daily historical data
    stock_data = yf.download(ticker_symbol, period='1y')

    # 2. Calculate the Simple Moving Average (SMA)
    # We will use a 20-day window for the SMA
    sma_window = 20
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=sma_window).mean()

    # 3. Calculate the Exponential Moving Average (EMA)
    # We will also use a 20-day span for the EMA so we can compare them
    ema_window = 20
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=ema_window, adjust=False).mean()

    # 4. Plot the results
    plt.figure(figsize=(14, 7)) # Set the size of the chart

    # Plot Closing Price, SMA, and EMA
    plt.plot(stock_data.index, stock_data['Close'], label=f'{ticker_symbol} Close Price', color='dodgerblue', alpha=0.6, linewidth=1.5)
    plt.plot(stock_data.index, stock_data['SMA_20'], label='20-Day SMA', color='orange', linestyle='--', linewidth=2)
    plt.plot(stock_data.index, stock_data['EMA_20'], label='20-Day EMA', color='crimson', linestyle='-.', linewidth=2)

    # Add titles and labels
    plt.title(f'{ticker_symbol} Stock Price with 20-Day SMA and EMA', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)

    # Add grid, legend, and format dates
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)
    plt.gcf().autofmt_xdate() # Automatically formats the dates on the x-axis nicely

    # 5. Display the chart
    print("Plotting the data...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
