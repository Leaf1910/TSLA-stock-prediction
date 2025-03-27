import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import yfinance as yf
from datetime import datetime, timedelta

# Constants
STARTING_CAPITAL = 10000  # USD
TRANSACTION_FEE = 0.01  # 1% transaction fee
MODEL_PATH = "trading_model.pkl"
FILE_PATH = "TSLA_updated.csv"
SIMULATION_DATES = pd.to_datetime(['2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28'])

# RSI calculation
def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Step 1: Fetch split-adjusted Tesla data
def fetch_tesla_data():
    print("Fetching updated Tesla stock data...")
    tsla = yf.download('TSLA', start='2010-06-29', end='2025-03-27')
    tsla.reset_index(inplace=True)
    tsla.to_csv(FILE_PATH, index=False)
    print(f"Data saved to {FILE_PATH}")
    return tsla

# Load or fetch data
try:
    data = pd.read_csv(FILE_PATH)
    data['Date'] = pd.to_datetime(data['Date'])
    if data['Date'].max() < SIMULATION_DATES[0]:
        data = fetch_tesla_data()
except FileNotFoundError:
    data = fetch_tesla_data()

# Step 2: Verify and clean data
print("First few rows of data:")
print(data.head())

# Ensure numeric columns
for col in ['Close', 'Open', 'Volume']:
    if not pd.api.types.is_numeric_dtype(data[col]):
        print(f"Converting '{col}' to numeric...")
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Feature engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['EMA_3'] = data['Close'].ewm(span=3, adjust=False).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['RSI'] = calculate_rsi(data)
data['Next_Open'] = data['Open'].shift(-1)
data.dropna(inplace=True)

# Define features and target
features = ['SMA_5', 'SMA_10', 'EMA_3', 'Daily_Return', 'RSI', 'Volume']
X = data[features]
y = data['Next_Open']

# Step 3: Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
joblib.dump(model, MODEL_PATH)
print(f"Model trained and saved at {MODEL_PATH}")

# Step 4: Simulate trading
capital = STARTING_CAPITAL
shares_held = 0
trade_log = []

for current_date in SIMULATION_DATES:
    if current_date > data['Date'].max():
        print(f"Simulation date {current_date} exceeds data range. Using latest data for prediction.")
        current_day = data.iloc[-1]
    else:
        current_day = data[data['Date'] == current_date].iloc[0]

    current_features = pd.DataFrame([current_day[features]], columns=features)
    current_price = float(current_day['Close'])
    prediction = float(model.predict(current_features)[0])

    # Debug thresholds
    buy_threshold = current_price * 1.002  # 0.2% increase
    sell_threshold = current_price * 0.998  # 0.2% decrease
    print(f"{current_date.strftime('%Y-%m-%d')} - Current: ${current_price:.2f}, Predicted: ${prediction:.2f}, Buy Threshold: ${buy_threshold:.2f}, Sell Threshold: ${sell_threshold:.2f}")

    # Trading decision
    order_action = "Hold"
    order_details = "No transaction"

    if prediction > buy_threshold and capital > 0:
        investment = min(capital * 0.5, capital)
        shares_to_buy = investment / current_price
        cost = shares_to_buy * current_price * (1 + TRANSACTION_FEE)
        capital -= cost
        shares_held += shares_to_buy
        order_action = "Buy"
        order_details = f"${investment:.2f}"
        print(f"{current_date.strftime('%Y-%m-%d')} 9:00 AM EST Submission: Buy: ${investment:.2f}")
    elif prediction < sell_threshold and shares_held > 0:
        proceeds = shares_held * current_price * (1 - TRANSACTION_FEE)
        capital += proceeds
        order_action = "Sell"
        order_details = f"{shares_held:.2f} shares"
        print(f"{current_date.strftime('%Y-%m-%d')} 9:00 AM EST Submission: Sell: {shares_held:.2f} shares")
        shares_held = 0
    else:
        print(f"{current_date.strftime('%Y-%m-%d')} 9:00 AM EST Submission: Hold")

    # Simulate execution
    if current_date < data['Date'].max():
        next_day = data[data['Date'] > current_date].iloc[0]
        execution_price = float(next_day['Open'])
        print(f"Executed at 10:00 AM EST: {order_action} at ${execution_price:.2f}")
    else:
        execution_price = prediction
        print(f"Executed at 10:00 AM EST (predicted): {order_action} at ${execution_price:.2f}")

    trade_log.append([current_date.strftime('%Y-%m-%d'), order_action, order_details])

# Step 5: Calculate final balance
final_price = float(data.iloc[-1]['Close']) if SIMULATION_DATES[-1] > data['Date'].max() else float(data[data['Date'] == SIMULATION_DATES[-1]]['Close'].iloc[0])
final_balance = capital + (shares_held * final_price)
performance = (final_balance - STARTING_CAPITAL) / STARTING_CAPITAL * 100
print(f"\nFinal Account Balance: ${final_balance:.2f}")
print(f"Performance over the period: {performance:.2f}%")

# Step 6: Save results
results_df = pd.DataFrame([{"Initial Capital": STARTING_CAPITAL, "Final Account Balance": final_balance, "Performance (%)": performance}])
results_df.to_csv("trading_results.csv", index=False)
trade_log_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Details"])
trade_log_df.to_csv("trade_log.csv", index=False)
print("Results and trade log saved.")