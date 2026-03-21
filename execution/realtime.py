# import requests
# import pandas as pd

# def fetch_realtime_aapl(api_key):
#     """Fetches 1-minute intraday data for AAPL from Alpha Vantage."""
#     symbol = "AAPL"
#     interval = "1min"

#     print(f"Requesting real-time {interval} data for {symbol}...")

#     # The Alpha Vantage API endpoint
#     url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}'

#     try:
#         # Make the HTTP request
#         response = requests.get(url)
#         data = response.json()

#         # Alpha Vantage wraps the data in a dictionary key based on the interval
#         time_series_key = f'Time Series ({interval})'

#         # Check if the API returned an error or rate limit warning
#         if time_series_key not in data:
#             print("\nError: Could not retrieve data. API Response:")
#             print(data)
#             return None

#         # Extract the actual pricing data
#         raw_data = data[time_series_key]

#         # 1. Convert the JSON dictionary into a Pandas DataFrame
#         # orient='index' tells pandas to use the timestamps as row indices
#         df = pd.DataFrame.from_dict(raw_data, orient='index')

#         # 2. Clean up the column names
#         # Alpha Vantage returns ugly names like '1. open', '2. high'. Let's strip the numbers.
#         df.columns = [col.split('. ')[1].capitalize() for col in df.columns]

#         # 3. Convert the data types
#         # The API returns everything as strings. We need floats for prices and ints for volume,
#         # and datetime objects for the index.
#         df.index = pd.to_datetime(df.index)
#         df = df.astype(float) # Convert all columns to floats
#         df['Volume'] = df['Volume'].astype(int) # Make Volume an integer

#         # 4. Sort the DataFrame so the newest data is at the top (or bottom, your choice)
#         df.sort_index(ascending=False, inplace=True)

#         return df

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# def main():
#     # --- INSERT YOUR API KEY HERE ---
#     # Fetch the API key securely from environment variables instead of hardcoding
#     import os
#     from dotenv import load_load
#     load_dotenv()
#     API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_API_KEY_HERE")

#     # Fetch the data
#     df = fetch_realtime_aapl(API_KEY)

#     # Print the most recent 5 minutes of data
#     if df is not None:
#         print("\n--- Latest AAPL Real-Time Data ---")
#         print(df.head(5))

# if __name__ == "__main__":
#     main()




# no need for api

import yfinance as yf
import pandas as pd
import time
from datetime import datetime

def fetch_latest_aapl_data():
    """Fetches the most recent 1-minute interval data for AAPL using yfinance."""
    ticker = 'AAPL'

    try:
        # Download the last 1 day of data, at a 1-minute interval
        # yfinance allows 1m intervals for a max of 7 days back.
        df = yf.download(ticker, period='1d', interval='1m', progress=False)

        # Check if the dataframe is empty (e.g., outside market hours)
        if df.empty:
            return None

        # Safeguard for the MultiIndex formatting issue we saw earlier
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # We just want the most recent row (the last minute)
        latest_data = df.iloc[-1]

        # Get the timestamp for this row
        timestamp = df.index[-1]

        return {
            'timestamp': timestamp,
            'open': latest_data['Open'],
            'high': latest_data['High'],
            'low': latest_data['Low'],
            'close': latest_data['Close'],
            'volume': latest_data['Volume']
        }

    except Exception as e:
        print(f"\nError fetching data: {e}")
        return None

def main():
    print("--- Starting Live AAPL Price Feed ---")
    print("Press Ctrl+C to stop the script.\n")

    # Keep track of the last timestamp we printed so we don't spam the console
    # with the exact same minute's data if we poll too fast.
    last_printed_time = None

    try:
        while True:
            current_time = datetime.now().strftime('%H:%M:%S')

            # Fetch the data
            data = fetch_latest_aapl_data()

            if data:
                data_time = data['timestamp']

                # Only print if we have a new 1-minute candle
                if data_time != last_printed_time:
                    print(f"[{current_time}] AAPL (1m Candle ending {data_time.strftime('%H:%M')}): "
                          f"Open: ${data['open']:.2f} | "
                          f"High: ${data['high']:.2f} | "
                          f"Low: ${data['low']:.2f} | "
                          f"Close: ${data['close']:.2f} | "
                          f"Volume: {int(data['volume'])}")

                    last_printed_time = data_time
                else:
                    # Optional: Print a dot to show the script is still alive and waiting
                    print(".", end="", flush=True)
            else:
                print(f"[{current_time}] No data returned. (Market closed?)")

            # Wait for 10 seconds before asking Yahoo Finance again.
            # You don't want to ping them every 0.1 seconds or they will block your IP.
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nLive feed stopped by user. Exiting...")

if __name__ == "__main__":
    main()
