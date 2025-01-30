import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error


def train_lstm(data):
    """
    Train an LSTM model for price prediction.
    
    Args:
        data (pd.DataFrame): Processed data with features.
    
    Returns:
        Sequential: Trained LSTM model.
    """
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(
        data[['price', 'price_lag1', 'price_rolling_avg', 'temperature', 'rain', 'sentiment_score']])

    # Prepare sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            # Use all features except the target
            X.append(data[i:i+seq_length, :-1])
            y.append(data[i+seq_length, 0])      # Target is the price
        return np.array(X), np.array(y)

    seq_length = 30  # Use 30 time steps for prediction
    X, y = create_sequences(data_scaled, seq_length)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(
        X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test))

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'LSTM MAE: {mae:.2f}')

    # Save model in .keras format
    model.save('lstm_model.keras')

    return model
