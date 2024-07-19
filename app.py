import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA

# https://blackschole.streamlit.app
# python -m streamlit run app.py

tab1, tab2, tab3 = st.tabs(["Stock Ticker Analysis", "Black Scholes Pricer", "Moving Averages"])

# Function to load data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Sidebar for user input
ticker = st.sidebar.text_input("Stock Ticker", value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Load data
data = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

with tab1:
    # App title
    st.title("Financial Calculator")
    
    # Display data
    st.subheader(f"{ticker} Stock Data")
    st.write(data.tail())

    # Plot data
    st.subheader(f"{ticker} Closing Price")
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    st.pyplot(fig)

with tab3:
    # Moving Averages
    st.subheader("Moving Averages")
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label='Closing Price')
    ax.plot(data['Date'], data['SMA_20'], label='20-Day SMA')
    ax.plot(data['Date'], data['SMA_50'], label='50-Day SMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# ARIMA Model
# st.subheader("ARIMA Model")
# model = ARIMA(data['Close'], order=(5, 1, 0))
# model_fit = model.fit(disp=0)
# data['ARIMA'] = model_fit.fittedvalues

# fig, ax = plt.subplots()
# ax.plot(data['Date'], data['Close'], label='Closing Price')
# ax.plot(data['Date'], data['ARIMA'], label='ARIMA (5,1,0)')
# ax.set_xlabel('Date')
# ax.set_ylabel('Price')
# ax.legend()
# st.pyplot(fig)