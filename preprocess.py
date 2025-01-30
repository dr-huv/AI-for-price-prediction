import pandas as pd


def load_and_preprocess_data(price_path, weather_path, trends_path):
    """
    Load and preprocess data for LSTM-based price prediction.
    
    Args:
        price_path (str): Path to commodity price data.
        weather_path (str): Path to weather data.
        trends_path (str): Path to market trends data.
    
    Returns:
        pd.DataFrame: Processed data with features.
    """
    # Load data
    price_data = pd.read_csv(price_path, parse_dates=[
                             'date'], index_col='date')
    weather_data = pd.read_csv(weather_path, parse_dates=[
                               'date'], index_col='date')
    trends_data = pd.read_csv(trends_path, parse_dates=[
                              'date'], index_col='date')

    # Merge data
    data = pd.merge(price_data, weather_data, on='date', how='left')
    data = pd.merge(data, trends_data, on='date', how='left')
    data.ffill(inplace=True)  # Forward fill missing values

    # Feature engineering
    data['price_lag1'] = data['price'].shift(1)  # Lagged price
    data['price_rolling_avg'] = data['price'].rolling(
        window=7).mean()  # Rolling average
    data['rain'] = data['rainfall'].apply(
        lambda x: 1 if x > 0 else 0)  # Binary rain feature
    data.dropna(inplace=True)  # Drop rows with NaN values

    return data
