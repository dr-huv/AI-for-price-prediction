import pandas as pd
import numpy as np

# Generate synthetic commodity price data
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
prices = np.random.normal(loc=100, scale=10, size=len(
    dates)).cumsum()  # Simulated price trend
price_data = pd.DataFrame({'date': dates, 'price': prices})

# Generate synthetic weather data
temperature = np.random.normal(loc=25, scale=5, size=len(dates))
# Simulated rainfall (count)
rainfall = np.random.poisson(lam=2, size=len(dates))
weather_data = pd.DataFrame(
    {'date': dates, 'temperature': temperature, 'rainfall': rainfall})

# Generate synthetic market trends data
sentiment_score = np.random.uniform(
    low=-1, high=1, size=len(dates))  # Simulated sentiment scores
trends_data = pd.DataFrame({'date': dates, 'sentiment_score': sentiment_score})

# Save datasets to CSV
price_data.to_csv('commodity_prices.csv', index=False)
weather_data.to_csv('weather_data.csv', index=False)
trends_data.to_csv('market_trends.csv', index=False)
