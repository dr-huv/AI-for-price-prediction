from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress all logs except errors
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs


# Load LSTM model
try:
    model_lstm = load_model('lstm_model.keras')  # Load .keras format
    print("LSTM model loaded successfully!")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")
    st.stop()

# Load data
data = pd.read_csv('processed_data.csv')

# Streamlit app
st.title('AI for Price Prediction Dashboard (LSTM)')

# Sidebar for user input
st.sidebar.header('User Input')
price_lag1 = st.sidebar.number_input('Price Lag 1', value=100.0)
price_rolling_avg = st.sidebar.number_input(
    'Price Rolling Average', value=105.0)
temperature = st.sidebar.number_input('Temperature', value=25.0)
rain = st.sidebar.selectbox('Rain', [0, 1])
sentiment_score = st.sidebar.number_input('Sentiment Score', value=0.8)

# Prepare input for LSTM
input_data = [[price_lag1, price_rolling_avg,
               temperature, rain, sentiment_score]]
input_data_scaled = np.array(input_data).reshape(1, -1)
input_data_scaled = np.repeat(input_data_scaled, 30, axis=0).reshape(
    1, 30, -1)  # Reshape for LSTM

# Make prediction
try:
    prediction_lstm = model_lstm.predict(input_data_scaled)[0][0]
    st.header('Prediction')
    st.write(f'LSTM Prediction: **{prediction_lstm:.2f}**')
except Exception as e:
    st.error(f"Error making prediction: {e}")

# Plot actual vs predicted prices
st.header('Actual vs Predicted Prices')
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['price'], label='Actual Prices')

# Prepare data for LSTM predictions
try:
    # Scale the data (using the same scaler as in training)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(
        data[['price', 'price_lag1', 'price_rolling_avg', 'temperature', 'rain', 'sentiment_score']])

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            # Use all features except the target
            X.append(data[i:i+seq_length, :-1])
            y.append(data[i+seq_length, 0])      # Target is the price
        return np.array(X), np.array(y)

    seq_length = 30
    X, y = create_sequences(data_scaled, seq_length)

    # Make predictions
    predictions = model_lstm.predict(X)

    # Plot predictions
    plt.plot(data.index[seq_length:], predictions, label='LSTM Predictions')
except Exception as e:
    st.error(f"Error generating predictions: {e}")

plt.legend()
plt.title('Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(plt)
