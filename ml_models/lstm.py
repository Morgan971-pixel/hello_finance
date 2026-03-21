import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def main():
    # 1. Fetch Historical Data
    ticker = 'AAPL'
    print(f"Fetching historical data for {ticker}...")
    # Fetching 4 years of data to give the neural network enough to learn from
    df = yf.download(ticker, start='2020-01-01', end='2024-01-01')

    # We only need the 'Close' price for this model
    data = df['Close'].values.reshape(-1, 1)

    # 2. Preprocess the Data (MinMax Scaling)
    # Neural networks perform much better when data is scaled between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Calculate the row to split the train/test sets (80% for training)
    training_data_len = int(np.ceil(len(data) * 0.8))

    # 3. Create the Training Dataset
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    # Lookback window: We will use the past 60 days to predict the next 1 day
    lookback = 60

    for i in range(lookback, len(train_data)):
        x_train.append(train_data[i-lookback:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data: LSTMs expect a 3D input [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 4. Build the LSTM Neural Network Model
    print("Building and training the LSTM model (this may take a minute)...")
    model = Sequential()

    # Add a first LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

    # Add a second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))

    # Add standard Dense layers to output the final prediction
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    # 5. Create the Testing Dataset
    # We need the last 60 days of the training set to predict the first day of the test set
    test_data = scaled_data[training_data_len - lookback: , :]

    x_test = []
    y_test = data[training_data_len:, :] # Actual unscaled closing prices for comparison

    for i in range(lookback, len(test_data)):
        x_test.append(test_data[i-lookback:i, 0])

    # Convert to numpy array and reshape for the LSTM
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 6. Make Predictions
    print("Making predictions on the test data...")
    predictions = model.predict(x_test)

    # Reverse the scaling to get actual dollar prices back
    predictions = scaler.inverse_transform(predictions)

    # Calculate the Root Mean Squared Error (RMSE) to evaluate accuracy
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f"Model RMSE: {rmse:.2f}")

    # 7. Visualize the Data
    train = df[:training_data_len].copy()
    valid = df[training_data_len:].copy()
    valid['Predictions'] = predictions

    plt.figure(figsize=(14, 7))
    plt.title('LSTM Neural Network - AAPL Stock Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)

    plt.plot(train.index, train['Close'], label='Training Data', color='dodgerblue')
    plt.plot(valid.index, valid['Close'], label='Actual Test Data', color='lightgray', linewidth=2)
    plt.plot(valid.index, valid['Predictions'], label='LSTM Predictions', color='crimson', linestyle='--')

    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
