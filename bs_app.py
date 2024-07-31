import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from black_scholes import Black_Scholes_Pricing

# Page Setup
st.set_page_config(
    page_title="Black-Scholes Options Pricing Model",
    layout="wide",
)

# Sidebar Title and LinkedIn Hyperlink
st.sidebar.title("Black-Scholes Model")

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

st.title("Black-Scholes Options Pricing Model")

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

# Generate Greek DataFrames
call_greek_df = pd.DataFrame({
            "Delta (δ)": [call_option.calc_delta()],
            "Gamma (γ)" : [call_option.calc_gamma()],
            "Theta (θ)" : [call_option.calc_theta()],
            "Vega (ν)" : [call_option.calc_vega()],
            "Rho (ρ)" : [call_option.calc_rho()]
        })

put_greek_df = pd.DataFrame({
            "Delta (δ)": [put_option.calc_delta()],
            "Gamma (γ)" : [put_option.calc_gamma()],
            "Theta (θ)" : [put_option.calc_theta()],
            "Vega (ν)" : [put_option.calc_vega()],
            "Rho (ρ)" : [put_option.calc_rho()]
        })

# HTML and CSS for the text boxes
call_box_html = f"""
<div style="border-radius: 15px; background-color: #5DADE2; padding: 20px; height: 100px; display: flex; flex-direction: column; justify-content: center; align-items: center; margin-bottom: 50px;">
    <h4 style="font-size: 18px; margin: 0; text-align: center;">Call Price</h4>
    <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
        <p style="font-size: 32px; font-weight: bold; margin: 0; text-align: center;">${call_price:.2f}</p>
    </div>
</div>
"""

put_box_html = f"""
<div style="border-radius: 15px; background-color: #FFA500; padding: 20px; height: 100px; display: flex; flex-direction: column; justify-content: center; align-items: center; margin-bottom: 50px;">
    <h4 style="font-size: 18px; margin: 0; text-align: center;">Put Price</h4>
    <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
        <p style="font-size: 32px; font-weight: bold; margin: 0; text-align: center;">${put_price:.2f}</p>
    </div>
</div>
"""

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
    st.write(call_box_html, unsafe_allow_html=True)
    st.dataframe(call_greek_df, use_container_width=True)
    st.plotly_chart(fig_call, use_container_width=True)
with col2:
    st.write(put_box_html, unsafe_allow_html=True)
    st.dataframe(put_greek_df, use_container_width=True)
    st.plotly_chart(fig_put, use_container_width=True)