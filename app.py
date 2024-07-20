import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go

# https://blackschole.streamlit.app
# python -m streamlit run app.py

# Page Setup
st.set_page_config(
    page_title="Financial Analysis App",
    layout="wide",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Stock Ticker Analysis", "Black-Scholes Model", "Signal Processing"])

# Introduction Page
if page == "Introduction":
    st.title("Introduction")
    st.write("Welcome to the Financial Analysis App. This application allows you to perform various financial analyses including stock ticker browsing, options pricing using the Black-Scholes model, and signal processing techniques. Use the navigation menu to explore the different functionalities.")

# Stock Ticker Analysis Page
elif page == "Stock Ticker Analysis":
    st.title("Stock Ticker Analysis")

    # Sidebar Inputs
    st.sidebar.subheader("Stock Ticker Options")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    start_date = st.sidebar.date_input("Start Date", dt.date.today() - dt.timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", dt.date.today())
    price_info = st.sidebar.selectbox("Information", ["Open", "High", "Low", "Close", "Adj Close", "Volume"], index=4)

    # Moving Average Options
    st.sidebar.subheader("Moving Average Options")
    sma_checkbox = st.sidebar.checkbox("Simple Moving Average")
    if sma_checkbox:
        sma_days = st.sidebar.number_input("Number of Days for SMA", min_value=1, max_value=365, value=30)

    ema_checkbox = st.sidebar.checkbox("Exponential Moving Average")
    if ema_checkbox:
        ema_days = st.sidebar.number_input("Number of Days for EMA", min_value=1, max_value=365, value=30)

    # Fetch Data
    def load_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        return data

    data = load_data(ticker, start_date, end_date)

    # Display Today's Prices in a Table
    st.subheader(f"{ticker} Stock Data for {dt.date.today()}" )
    today_data = data.iloc[-1]
    today_df = pd.DataFrame({
        "Open": [today_data['Open']],
        "High": [today_data['High']],
        "Low": [today_data['Low']],
        "Close": [today_data['Close']],
        "Adj Close": [today_data['Adj Close']],
        "Volume": [today_data['Volume']]
    })
    st.write(today_df.to_html(index=False), unsafe_allow_html=True)

    # Stock Ticker Graph with Moving Averages
    # st.subheader("Stock Ticker")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[price_info], mode='lines', name=price_info))

    if sma_checkbox:
        data['SMA'] = data[price_info].rolling(window=sma_days).mean()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA'], mode='lines', name=f"{sma_days}-Day SMA"))

    if ema_checkbox:
        data['EMA'] = data[price_info].ewm(span=ema_days, adjust=False).mean()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA'], mode='lines', name=f"{ema_days}-Day EMA"))

    fig.update_layout(title=f"{ticker} {price_info} Price",
                      xaxis_title="Date",
                      yaxis_title=price_info,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))

    st.plotly_chart(fig, use_container_width=True)

    # Placeholder for ARIMA and GARCH Models
    st.subheader("ARIMA Model")
    st.write("To be filled in later.")

    st.subheader("GARCH Model")
    st.write("To be filled in later.")

# Black-Scholes Model Page
elif page == "Black-Scholes Model":
    st.title("Black-Scholes Model")
    # Placeholder for Black-Scholes Model implementation
    st.write("This section will be implemented later.")

# Signal Processing Page
elif page == "Signal Processing":
    st.title("Signal Processing")

    # Placeholder for Signal Processing implementation
    st.subheader("Fourier Transform")
    st.write("To be filled in later.")

    st.subheader("Wavelet Transform")
    st.write("To be filled in later.")

    st.subheader("Kalman Filter")
    st.write("To be filled in later.")