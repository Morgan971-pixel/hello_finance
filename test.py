import yfinance as yf
import matplotlib.pyplot as plt

# 1. Fetch Apple (AAPL) historical data
# The period='5y' fetches the last 5 years of data.
# Alternatively, you can specify dates: yf.download('AAPL', start='2020-01-01', end='2024-01-01')
print("Fetching Apple stock data...")
aapl_data = yf.download('AAPL', period='5y')

# 2. Plot the 'Close' price using matplotlib
plt.figure(figsize=(12, 6)) # Set the figure size to make it readable

# Create the line chart
plt.plot(aapl_data.index, aapl_data['Close'], label='AAPL Close Price', color='royalblue', linewidth=1.5)

# Add title and labels
plt.title('Apple (AAPL) Historical Stock Price', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price (USD)', fontsize=12)

# Add grid lines and a legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)

# Automatically format the x-axis dates to look nice
plt.gcf().autofmt_xdate()

# 3. Display the chart
plt.show()
