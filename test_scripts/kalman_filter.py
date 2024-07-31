import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from pykalman import KalmanFilter

# Function to fetch data from earliest available date to current date
def load_data_full(ticker):
    data = yf.download(ticker, period="max")
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}.")
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Get ticker data
ticker = "AAPL"
data_full = load_data_full(ticker)

# Generate Simple Moving Average
sma_days = 30
data_full['SMA'] = data_full['Adj Close'].rolling(window=sma_days).mean()

# Generate Exponential Moving Average
ema_days = 30
data_full['EMA'] = data_full['Adj Close'].ewm(span=ema_days, adjust=False).mean()

# Generate Kalman Filter Time Series
kf = KalmanFilter(transition_matrices = [1],
              observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.0001)

kf_mean, kf_cov = kf.filter(data_full['Adj Close'].values)
data_full['Kalman'] = kf_mean.flatten()

# Create and display figure
fig = go.Figure()

fig.add_trace(go.Scatter(x=data_full['Date'], y=data_full['Adj Close'], mode='lines', name='Adj Close'))
fig.add_trace(go.Scatter(x=data_full['Date'], y=data_full['SMA'], mode='lines', name=f"{sma_days}-Day SMA"))
fig.add_trace(go.Scatter(x=data_full['Date'], y=data_full['EMA'], mode='lines', name=f"{ema_days}-Day EMA"))
fig.add_trace(go.Scatter(x=data_full['Date'], y=data_full['Kalman'], mode='lines', name="Kalman Filter"))

fig.update_layout(title=f"{ticker} Adj Close Price",
                          xaxis=dict(range=[dt.datetime.today() - dt.timedelta(days=365), dt.datetime.today()]),
                          xaxis_title="Date",
                          yaxis_title='Adj Close',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))

fig.show()