import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import timedelta

def main():
    # 1. Fetch Historical Data
    print("Fetching historical data for AAPL...")
    df = yf.download('AAPL', period='2y') # Let's look at the last 2 years
    df.reset_index(inplace=True) # Move the 'Date' index into a standard column

    # 2. Prepare the Features (X) and Target (y)
    # Machine learning models need numbers, not date objects.
    # We will convert the dates into "Days since the start of the dataset"
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days

    # Reshape X to be a 2D array, which scikit-learn requires
    X = df[['Days']].values
    y = df['Close'].values

    # 3. Split the Data into Training and Testing Sets
    # We will use 80% of the data to train the model, and 20% to test its accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and Train the Linear Regression Model
    print("Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate how well the model fit the testing data (R-squared score)
    # A score closer to 1.0 is better. (Expect this to be low for stocks!)
    test_score = model.score(X_test, y_test)
    print(f"Model R^2 Score on Test Data: {test_score:.4f}")

    # 5. Predict the Next 7 Days
    print("\n--- Predicted Closing Prices for the Next 7 Days ---")

    last_day_integer = df['Days'].max()
    last_actual_date = df['Date'].max()

    # Create the feature arrays for the next 7 days
    future_days = np.array([[last_day_integer + i] for i in range(1, 8)])

    # Get the calendar dates for those future days
    future_dates = [last_actual_date + timedelta(days=i) for i in range(1, 8)]

    # Make the predictions and flatten the array to 1D
    future_predictions = model.predict(future_days).flatten()

    # Print the results
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

    # 6. Visualize the Data and the Regression Line
    plt.figure(figsize=(12, 6))

    # Plot the actual historical prices
    plt.scatter(df['Date'], df['Close'], color='dodgerblue', label='Actual Prices', s=10, alpha=0.6)

    # Plot the straight line that the Linear Regression model learned
    plt.plot(df['Date'], model.predict(X), color='red', label='Linear Regression Line', linewidth=2)

    # Plot the 7 future predicted points
    plt.scatter(future_dates, future_predictions, color='limegreen', label='7-Day Prediction', zorder=5, s=50)

    plt.title('AAPL Stock Price Prediction using Linear Regression', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
