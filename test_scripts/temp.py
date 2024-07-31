import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA

from black_scholes import Black_Scholes_Pricing

# python -m streamlit run app.py

# Page Setup
st.set_page_config(
    page_title="Financial Analysis App",
    layout="wide",
)

# Sidebar Title and LinkedIn Hyperlink
st.sidebar.title("Financial Analysis")

linkedin_url = "https://www.linkedin.com/in/karthik-selvaraj-purdue/"
linkedin_html = f"""
<div style="display: flex; align-items: center;  margin-bottom: 20px;">
    <span style='font-size:14px; margin-right: 5px;'>Developed by </span>
    <a href="{linkedin_url}" target="_blank" style="display: flex; align-items: center; text-decoration: none;">
        <span style='font-size:14px;'>Karthik Selvaraj</span>
    </a>
</div>
"""
st.sidebar.markdown(linkedin_html, unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Stock Ticker Analysis", "Options Pricing", "Volatility Modeling", "Signal Processing"])

# Function to fetch data and ensure proper indexing
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} from {start_date} to {end_date}.")
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Function to fetch data from earliest available date to current date
def load_data_full(ticker):
    data = yf.download(ticker, period="max")
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}.")
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Introduction Page
# if page == "Introduction":
#     st.title("Introduction")
#     st.write("Welcome to the Financial Analysis App. This application allows you to perform various financial analyses including stock ticker browsing, options pricing using the Black-Scholes model, and signal processing techniques. Use the navigation menu to explore the different functionalities.")

# Stock Ticker Analysis Page
if page == "Stock Ticker Analysis":
    st.title("Stock Ticker Analysis")

    # Sidebar Inputs
    st.sidebar.markdown("""---""")
    st.sidebar.subheader("Stock Ticker Options")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    start_date = st.sidebar.date_input("Start Date", dt.date.today() - dt.timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", dt.date.today())
    price_info = st.sidebar.selectbox("Information", ["Open", "High", "Low", "Close", "Adj Close", "Volume"], index=4)

    # Moving Average Options
    st.sidebar.markdown("""---""")
    st.sidebar.subheader("Moving Average Options")
    sma_checkbox = st.sidebar.checkbox("Simple Moving Average")
    if sma_checkbox:
        sma_days = st.sidebar.number_input("Number of Days for SMA", min_value=1, max_value=365, value=30)

    ema_checkbox = st.sidebar.checkbox("Exponential Moving Average")
    if ema_checkbox:
        ema_days = st.sidebar.number_input("Number of Days for EMA", min_value=1, max_value=365, value=30)

    # ARIMA Options
    st.sidebar.markdown("""---""")
    st.sidebar.subheader("ARIMA Options")
    steps = st.sidebar.number_input("Steps for Prediction", min_value=1, max_value=365, value=30)
    window_view = st.sidebar.number_input("Window View (days)", min_value=1, max_value=365, value=30)

    try:
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

        # Fetch full data for ARIMA model
        data_full = load_data_full(ticker)

        # Filter data for the window view
        view_start_date = data_full['Date'].iloc[-1] - pd.Timedelta(days=window_view)
        data_view = data_full[data_full['Date'] >= view_start_date]

        # ARIMA Model Section
        st.subheader("ARIMA Model")
        try:
            # Check for stationarity using ADF test
            result = adfuller(data_full['Adj Close'])
            st.write(f"ADF Statistic: {result[0]}")
            st.write(f"p-value: {result[1]}")
            for key, value in result[4].items():
                st.write(f'Critical Values {key}: {value}')

            d = 0
            if result[1] > 0.05:
                st.write("The series is non-stationary and needs differencing.")
                d = 1

            # Determine p and q using PACF and ACF
            lag_acf = acf(data_full['Adj Close'].diff(d).dropna(), nlags=20)
            lag_pacf = pacf(data_full['Adj Close'].diff(d).dropna(), nlags=20, method='ols')

            p = np.argmax(lag_pacf < 0.05)
            q = np.argmax(lag_acf < 0.05)

            st.write(f"Suggested p value: {p}")
            st.write(f"Suggested d value: {d}")
            st.write(f"Suggested q value: {q}")

            model_arima = ARIMA(data_full['Adj Close'], order=(p, d, q)).fit()
            arima_predictions = model_arima.forecast(steps=steps)

            # Plot ARIMA Predictions
            arima_fig = go.Figure()
            arima_fig.add_trace(go.Scatter(x=data_view['Date'], y=data_view['Adj Close'], mode='lines', name='Actual'))
            arima_fig.add_trace(go.Scatter(x=pd.date_range(start=data_full['Date'].iloc[-1], periods=steps + 1, freq='B')[1:], y=arima_predictions, mode='lines', name='ARIMA Predictions'))
            arima_fig.update_layout(title="ARIMA Model Predictions", xaxis_title="Date", yaxis_title="Price", legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
            st.plotly_chart(arima_fig, use_container_width=True)
        except Exception as e:
            st.write(f"Error in ARIMA model: {e}")

    except ValueError as e:
        st.error(e)
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Options Pricing Page
elif page == "Options Pricing":
    st.title("Options Pricing")
    tabs = st.tabs(["Black-Scholes", "Binomial"])

    with tabs[0]:
        st.header("Black-Scholes Model")

        # Sidebar Inputs
        st.sidebar.markdown("""---""")
        st.sidebar.subheader("Black-Scholes Model Inputs")
        spot_price = st.sidebar.number_input("Spot Price (S)", value=100.00, format="%.2f", min_value=0.00)
        strike_price = st.sidebar.number_input("Strike Price (K)", value=110.00, format="%.2f", min_value=0.00)
        days_to_maturity = st.sidebar.number_input("Days to Maturity (t)", value=365, format="%d", min_value=0)
        risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, format="%.2f", min_value=0.00)
        dividends = st.sidebar.number_input("Dividends (q)", value=0.00, format="%.2f", min_value=0.00)
        volatility = st.sidebar.number_input("Volatility (σ)", value=0.25, format="%.2f")

        st.sidebar.markdown("""---""")
        st.sidebar.subheader("Heatmap Inputs")
        min_spot = st.sidebar.number_input("Min Spot Price", value=75.00, format="%.2f", min_value=0.00)
        max_spot = st.sidebar.number_input("Max Spot Price", value=125.00, format="%.2f", min_value=0.00)
        min_vol = st.sidebar.number_input("Min Volatility", value=0.01, format="%.2f")
        max_vol = st.sidebar.number_input("Max Volatility", value=1.00, format="%.2f")
        granularity = st.sidebar.slider("Granularity", value=10, format="%d", min_value=5, max_value=20)

        # Calculate values for Call and Put prices
        call_option = Black_Scholes_Pricing(spot_price, strike_price, days_to_maturity, risk_free_rate, dividends, volatility, "call")
        put_option = Black_Scholes_Pricing(spot_price, strike_price, days_to_maturity, risk_free_rate, dividends, volatility, "put")

        call_price = call_option.price
        put_price = put_option.price

        # HTML and CSS for the table and text boxes
        html_content = f"""
        <style>
            .disable-svg svg {{
                display: none;
            }}
        </style>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="disable-svg" style="flex: 1; text-align: center; margin-right: 12px;">
                <div style="border-radius: 15px; background-color: #5DADE2; padding: 20px; height: 100px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <h4 style="font-size: 18px; margin: 0; text-align: center;">Call Price</h4>
                    <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                        <p style="font-size: 32px; font-weight: bold; margin: 0; text-align: center;">${call_price:.2f}</p>
                    </div>
                </div>
            </div>
            <div class="disable-svg" style="flex: 1; text-align: center; margin-left: 12px;">
                <div style="border-radius: 15px; background-color: #FFA500; padding: 20px; height: 100px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <h4 style="font-size: 18px; margin: 0; text-align: center;">Put Price</h4>
                    <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                        <p style="font-size: 32px; font-weight: bold; margin: 0; text-align: center;">${put_price:.2f}</p>
                    </div>
                </div>
            </div>
        </div>
        """

        # Display the HTML content
        st.write(html_content, unsafe_allow_html=True)

        # Generates Heatmaps
        call_grid, call_heat_spots, call_heat_vols = call_option.gen_heatmap(min_spot, max_spot, min_vol, max_vol, gran=granularity)
        put_grid, put_heat_spots, put_heat_vols = put_option.gen_heatmap(min_spot, max_spot, min_vol, max_vol, gran=granularity)

        # Custom color scale
        colorscale = [
            [0, 'rgb(255,0,0)'], # Red
            [1, 'rgb(0,255,0)']  # Green
        ]

        # Create separate figures for call and put heatmaps with annotations
        fig_call = go.Figure(data=go.Heatmap(
            z=call_grid,
            x=call_heat_spots,
            y=call_heat_vols,
            text=call_grid,
            texttemplate="%{text:.2f}",
            colorscale=colorscale
        ))
        fig_call.update_layout(
            title=dict(text="Call Heatmap", x=0.5, xanchor='center', font=dict(size=24)),
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            autosize=True,
            height=800,
            width=600
        )

        fig_put = go.Figure(data=go.Heatmap(
            z=put_grid,
            x=put_heat_spots,
            y=put_heat_vols,
            text=put_grid,
            texttemplate="%{text:.2f}",
            colorscale=colorscale
        ))
        fig_put.update_layout(
            title=dict(text="Put Heatmap", x=0.5, xanchor='center', font=dict(size=24)),
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            autosize=True,
            height=800,
            width=600
        )

        # Display the figures side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_call, use_container_width=True)
        with col2:
            st.plotly_chart(fig_put, use_container_width=True)


    with tabs[1]:
        st.header("Binomial Model")
        # Placeholder for Binomial Model implementation
        st.write("This section will be implemented later.")

# Volatility Modeling Page
elif page == "Volatility Modeling":
    st.title("Volatility Models")

    # Placeholders for ARCH, GARCH, and Jump Diffusion Models
    st.subheader("ARCH Model")
    st.write("To be filled in later.")

    st.subheader("GARCH Model")
    st.write("To be filled in later.")

    st.subheader("Jump Diffusion")
    st.write("To be filled in later.")

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