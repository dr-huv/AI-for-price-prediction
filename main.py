from preprocess import load_and_preprocess_data
from train import train_lstm

# Load and preprocess data
data = load_and_preprocess_data(
    'commodity_prices.csv', 'weather_data.csv', 'market_trends.csv')

# Train LSTM model
train_lstm(data)

# Save processed data
data.to_csv('processed_data.csv', index=False)
