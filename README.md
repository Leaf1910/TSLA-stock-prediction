*Trading Model*

*Introduction*
This repository includes a script to predict the next day's opening price of Tesla stock and simulate a simple trading strategy. Due to RAM constraints, this version uses a RandomForestRegressor with 200 estimators. These limitations may result in less precise predictions compared to using a larger dataset and/or more robust model configurations.

*File Overview*
1. TSLA_updated.csv
   - CSV file containing Tesla stock price data fetched from yfinance, including 'Close', 'Open', and 'Volume' columns used for predictions.
2. trading_model.pkl
   - Serialized trained RandomForestRegressor model saved for later use.
3. trading_results.csv
   - Output file summarizing the final trading performance, including initial capital, final balance, and percentage performance.
4. trade_log.csv
   - Output file logging daily trading decisions (Buy, Sell, Hold).

*How to Run*
1. Install Dependencies
   - Ensure you have Python 3+ installed, along with the required libraries:
     pip install pandas numpy scikit-learn yfinance joblib

2. Check File Paths
   - Ensure the paths for TSLA_updated.csv (stock data) and trading_model.pkl (saved model) are correct in main.py. By default, they are set to the project directory.

3. Execute the Script
   - Run the script in a Python environment (e.g., command line or Jupyter Notebook):
     python main.py
   - The code will print:
     - Data loading and cleaning steps
     - Model training confirmation
     - Daily trading decisions (e.g., "Buy: $5000.00", "Sell: 17.96 shares", "Hold")
     - Final account balance and trading performance

4. Output
   - A CSV file named trading_results.csv containing:
     - Initial Capital
     - Final Account Balance
     - Performance (%)
   - A CSV file named trade_log.csv containing daily trading actions.

*Code Sections (Brief Overview)*
1. Imports & Globals
   - Loads necessary libraries (pandas, numpy, scikit-learn, yfinance, joblib)
   - Defines file paths and constants (starting capital: $10,000, transaction fee: 1%)

2. Data Loading & Processing
   - Fetches TSLA data from yfinance (2010 to March 26, 2025)
   - Ensures 'Date' column is in datetime format
   - Calculates features: SMA_5, SMA_10, EMA_3, Daily_Return, RSI, and uses Volume
   - Drops missing values and shifts 'Open' to obtain 'Next_Open' for prediction

3. Model Training
   - Trains a RandomForestRegressor with 200 trees on all available data
   - Saves the trained model as trading_model.pkl

4. Trading Simulation
   - Uses a predefined date range for trading (2025-03-24 to 2025-03-28)
   - Predicts next day's opening price and executes buy/sell decisions:
     - Buy: If predicted price > current + 0.2%, invest 50% of capital
     - Sell: If predicted price < current - 0.2%, sell all shares
     - Hold: Otherwise
   - Calculates final account balance and performance
   - Saves results to trading_results.csv and trade_log.csv

*Limitations and Next Steps*
- Memory Constraints: This version is optimized for lower RAM usage. Expanding the dataset or increasing model complexity may require additional memory resources.
- Accuracy: Performance may be improved by:
  - Training on more years of data
  - Increasing n_estimators for the RandomForest model
  - Using advanced models like CatBoost or LightGBM
- Feature Engineering: Additional stock indicators (e.g., MACD) and advanced transformations may improve predictive accuracy.

*Example Output*
2025-03-24 - Current: $278.39, Predicted: $280.29, Buy Threshold: $278.95, Sell Threshold: $277.83
2025-03-24 9:00 AM EST Submission: Buy: $5000.00
Executed at 10:00 AM EST: Buy at $283.60
...
Final Account Balance: $10073.36
Performance over the period: 0.73%
