import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from black_scholes import Black_Scholes_Pricing
from monte_carlo import Monte_Carlo_Pricing
from binomial import Binomial_Pricing
from trinomial import Trinomial_Pricing

# Page Setup
st.set_page_config(
    page_title="Black-Scholes Options Pricing Model",
    layout="wide",
)

# Sidebar Title and LinkedIn Hyperlink
st.sidebar.title("Options Pricing")

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
page = st.sidebar.selectbox("Models", ["Black-Scholes Model", "Monte-Carlo Simulation", "Binomial Model", "Trinomial Model"], index=0)

def make_text_box(label, value, color, margin_bottom="50"):
    text_box_html = f"""
    <style>
        .disable-svg svg {{
            display: none;
        }}
    </style>
    <div class="disable-svg" style="border-radius: 15px; background-color: {color}; padding: 20px; height: 100px; display: flex; flex-direction: column; justify-content: center; align-items: center; margin-bottom: {margin_bottom}px;">
        <h4 style="font-size: 18px; margin: 0; text-align: center; margin-bottom: -15px; margin-top: -5px;">{label}</h4>
        <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
            <p style="font-size: 32px; font-weight: bold; margin: 0; text-align: center;">{value}</p>
        </div>
    </div>
    """

    return text_box_html

# Black Scholes Page
if page == "Black-Scholes Model":
    st.title("Black-Scholes Model")

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
    call_box_html = make_text_box("Call Price", f"${call_price:.2f}", "#5DADE2")
    put_box_html = make_text_box("Put Price", f"${put_price:.2f}", "#FFA500")

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



# Monte-Carlo Page
elif page == "Monte-Carlo Simulation":
    st.title("Monte-Carlo Pricer (Brownian Motion)")

    # Sidebar Inputs
    st.sidebar.markdown("""---""")
    st.sidebar.subheader("Monte-Carlo Inputs")
    spot_price = st.sidebar.number_input("Spot Price (S)", value=100.00, format="%.2f", min_value=0.00)
    strike_price = st.sidebar.number_input("Strike Price (K)", value=110.00, format="%.2f", min_value=0.00)
    days_to_maturity = st.sidebar.number_input("Days to Maturity (t)", value=365, format="%d", min_value=0)
    risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, format="%.2f", min_value=0.00)
    volatility = st.sidebar.number_input("Volatility (σ)", value=0.25, format="%.2f")
    iterations = st.sidebar.number_input("Number of Iterations", value=100, format="%d", min_value=1)

    st.sidebar.markdown("""---""")
    st.sidebar.subheader("VaR Inputs")
    format_var = st.sidebar.radio("Format", ["Dollar Amount", "Percentage"], index=0)
    confidence_level = st.sidebar.slider("VaR Confidence Level", value=0.950, format="%.3f", min_value=0.900, max_value=0.999, step=0.001)

    # Calculate values for Call and Put prices
    option = Monte_Carlo_Pricing(spot_price, strike_price, days_to_maturity, risk_free_rate, volatility, iterations)

    call_price = option.call_price
    put_price = option.put_price

    sim = option.S_n

    # Creates Simulation Graphs
    fig_sim = go.Figure()

    for i in range(iterations):
        fig_sim.add_trace(go.Scatter(x=np.arange(days_to_maturity + 1), y=sim[:, i], mode='lines', line=dict(color='rgb(3, 186, 255)', width=1), showlegend=False))

    fig_sim.add_trace(go.Scatter(x=np.arange(days_to_maturity + 1), y=[strike_price] * (days_to_maturity + 1), mode='lines', line=dict(color='rgb(255,0,0)', width=3), name='Strike Price'))

    fig_sim.update_layout(
        title=dict(text="Monte-Carlo Simulation", x=0.5, xanchor='center', font=dict(size=24)),
        xaxis_title="Days",
        yaxis_title="Simulated Price",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        autosize=True,
        height=800,
        width=600
    )

    # HTML and CSS for the text boxes
    call_box_html = make_text_box("Call Price", f"${call_price:.2f}", "#5DADE2")
    put_box_html = make_text_box("Put Price", f"${put_price:.2f}", "#FFA500")

    # Display the figure and prices side by side
    col1a, col2a = st.columns(2)
    with col1a:
        st.write(call_box_html, unsafe_allow_html=True)
        # st.dataframe(call_itm_df, use_container_width=True)
    with col2a:
        st.write(put_box_html, unsafe_allow_html=True)
        # st.dataframe(put_itm_df, use_container_width=True)

    st.plotly_chart(fig_sim, use_container_width=True)

    # Calculate VaR and CVaR values
    if format_var == "Dollar Amount":
        call_returns = np.sort(sim[-1] - spot_price)
        put_returns = np.sort(spot_price - sim[-1])

        call_VaR = np.percentile(call_returns, (1 - confidence_level) * 100)
        put_VaR = np.percentile(put_returns, (1 - confidence_level) * 100)

        call_CVaR = np.mean(call_returns[call_returns <= call_VaR])
        put_CVaR = np.mean(put_returns[put_returns <= put_VaR])

        call_VaR_sign = "-" if call_VaR < 0 else ""
        put_VaR_sign = "-" if put_VaR < 0 else ""
        call_CVaR_sign = "-" if call_CVaR < 0 else ""
        put_CVaR_sign = "-" if put_CVaR < 0 else ""

        str_call_VaR = f"{call_VaR_sign}${abs(call_VaR):,.2f}"
        str_put_VaR = f"{put_VaR_sign}${abs(put_VaR):,.2f}"
        str_call_CVaR = f"{call_CVaR_sign}${abs(call_CVaR):,.2f}"
        str_put_CVaR = f"{put_CVaR_sign}${abs(put_CVaR):,.2f}"

    elif format_var == "Percentage":
        call_returns = np.sort((sim[-1] - spot_price) / spot_price)
        put_returns = np.sort((spot_price - sim[-1]) / spot_price)

        call_VaR = np.percentile(call_returns, (1 - confidence_level) * 100) * 100
        put_VaR = np.percentile(put_returns, (1 - confidence_level) * 100) * 100

        call_CVaR = np.mean(call_returns[call_returns <= (call_VaR / 100)]) * 100
        put_CVaR = np.mean(put_returns[put_returns <= (put_VaR / 100)]) * 100

        str_call_VaR = f"{call_VaR:.2f}%"
        str_put_VaR = f"{put_VaR:.2f}%"
        str_call_CVaR = f"{call_CVaR:.2f}%"
        str_put_CVaR = f"{put_CVaR:.2f}%"

    # Generate ITM Probabilities
    call_itm_prob = np.mean(sim[-1] > strike_price)
    put_itm_prob = np.mean(sim[-1] < strike_price)

    # call_itm_df = pd.DataFrame({"Probability of Call Expiring ITM": ["{:.2%}".format(call_itm_prob)]})
    # put_itm_df = pd.DataFrame({"Probability of Put Expiring ITM": ["{:.2%}".format(put_itm_prob)]})

    call_itm_html = make_text_box("Probability Call Expires ITM", f"{call_itm_prob:.2%}", "#499140")
    put_itm_html = make_text_box("Probability Put Expires ITM", f"{put_itm_prob:.2%}", "#499140")

    # Create and display text boxes for VaR and CVar
    call_var_box_html = make_text_box("Call Value at Risk (VaR)", str_call_VaR, "#EB345E", margin_bottom="20")
    put_var_box_html = make_text_box("Put Value at Risk (VaR)", str_put_VaR, "#EB345E", margin_bottom="20")

    call_cvar_box_html = make_text_box("Call Conditional Value at Risk (CVaR)", str_call_CVaR, "#A434EB")
    put_cvar_box_html = make_text_box("Put Conditional Value at Risk (CVaR)", str_put_CVaR, "#A434EB")

    col1b, col2b = st.columns(2)
    with col1b:
        st.write(call_itm_html, unsafe_allow_html=True)
        st.write(call_var_box_html, unsafe_allow_html=True)
        st.write(call_cvar_box_html, unsafe_allow_html=True)
    with col2b:
        st.write(put_itm_html, unsafe_allow_html=True)
        st.write(put_var_box_html, unsafe_allow_html=True)
        st.write(put_cvar_box_html, unsafe_allow_html=True)

# Binomial Model
elif page == "Binomial Model":
    st.title("Binomial Pricing Model")

    # Sidebar Inputs
    st.sidebar.markdown("""---""")
    st.sidebar.subheader("Binomial Model Inputs")
    spot_price = st.sidebar.number_input("Spot Price (S)", value=100.00, format="%.2f", min_value=0.00)
    strike_price = st.sidebar.number_input("Strike Price (K)", value=110.00, format="%.2f", min_value=0.00)
    days_to_maturity = st.sidebar.number_input("Days to Maturity (t)", value=365, format="%d", min_value=0)
    risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, format="%.2f", min_value=0.00)
    volatility = st.sidebar.number_input("Volatility (σ)", value=0.25, format="%.2f")
    steps = st.sidebar.number_input("Number of Steps", value=10, format="%d", min_value=1)

    # Calculate call and put option prices
    option = Binomial_Pricing(spot_price, strike_price, days_to_maturity, risk_free_rate, volatility, steps)

    call_price = option.call_price
    put_price = option.put_price

    # Generate the binomial tree plot
    bin_fig = go.Figure()

    for i in range(steps + 1):
        for j in range(i + 1):
            if i > 0 and option.ST[i - 1, j] != 0:  # Avoid adding edges that go down to 0
                bin_fig.add_trace(go.Scatter(
                    x=[i - 1, i],
                    y=[option.ST[i - 1, j], option.ST[i, j]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))

            if i > 0 and j > 0 and option.ST[i - 1, j - 1] != 0:  # Avoid adding edges that go down to 0
                bin_fig.add_trace(go.Scatter(
                    x=[i - 1, i],
                    y=[option.ST[i - 1, j - 1], option.ST[i, j]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))

    # Add markers after lines to ensure they are on top
    for i in range(steps + 1):
        for j in range(i + 1):
            bin_fig.add_trace(go.Scatter(
                x=[i],
                y=[option.ST[i, j]],
                mode='markers',
                text=[f'{option.ST[i, j]:.2f}'],
                textposition='top center',
                marker=dict(size=10),
                showlegend=False
            ))

    bin_fig.update_layout(
        title=dict(text="Binomial Pricing Tree", x=0.5, xanchor='center', font=dict(size=24)),
        xaxis_title="Steps",
        yaxis_title="Price",
        showlegend=False,
        height=800,
        width=800
    )

    # HTML and CSS for the text boxes
    call_box_html = make_text_box("Call Price", f"${call_price:.2f}", "#5DADE2")
    put_box_html = make_text_box("Put Price", f"${put_price:.2f}", "#FFA500")

    # Display the prices and tree plot
    col1, col2 = st.columns(2)
    with col1:
        st.write(call_box_html, unsafe_allow_html=True)
    with col2:
        st.write(put_box_html, unsafe_allow_html=True)

    st.plotly_chart(bin_fig, use_container_width=True)

# Trinomial Model
elif page == "Trinomial Model":
    st.title("Trinomial Pricing Model")

    # Sidebar Inputs
    st.sidebar.markdown("""---""")
    st.sidebar.subheader("Trinomial Model Inputs")
    spot_price = st.sidebar.number_input("Spot Price (S)", value=100.00, format="%.2f", min_value=0.00)
    strike_price = st.sidebar.number_input("Strike Price (K)", value=110.00, format="%.2f", min_value=0.00)
    days_to_maturity = st.sidebar.number_input("Days to Maturity (t)", value=365, format="%d", min_value=0)
    risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, format="%.2f", min_value=0.00)
    volatility = st.sidebar.number_input("Volatility (σ)", value=0.25, format="%.2f")
    steps = st.sidebar.number_input("Number of Steps", value=10, format="%d", min_value=1)

    # Calculate call and put option prices
    option = Trinomial_Pricing(spot_price, strike_price, days_to_maturity, risk_free_rate, volatility, steps)

    call_price = option.call_price
    put_price = option.put_price

    # Generate the trinomial tree plot
    trin_fig = go.Figure()

    for i in range(steps + 1):
        for j in range(2 * i + 1):
            if i > 0:
                if option.ST[option.steps - i + j, i - 1] != 0 and option.ST[option.steps - i + j - 1, i] != 0:
                    trin_fig.add_trace(go.Scatter(
                        x=[i - 1, i],
                        y=[option.ST[option.steps - i + j, i - 1], option.ST[option.steps - i + j - 1, i]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ))
                if option.ST[option.steps - i + j, i - 1] != 0 and option.ST[option.steps - i + j + 1, i] != 0:
                    trin_fig.add_trace(go.Scatter(
                        x=[i - 1, i],
                        y=[option.ST[option.steps - i + j, i - 1], option.ST[option.steps - i + j + 1, i]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ))
                if option.ST[option.steps - i + j, i - 1] != 0 and option.ST[option.steps - i + j, i] != 0:
                    trin_fig.add_trace(go.Scatter(
                        x=[i - 1, i],
                        y=[option.ST[option.steps - i + j, i - 1], option.ST[option.steps - i + j, i]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ))

    # Add markers after lines to ensure they are on top
    for i in range(steps + 1):
        for j in range(2 * i + 1):
            if option.ST[option.steps - i + j, i] != 0:
                trin_fig.add_trace(go.Scatter(
                    x=[i],
                    y=[option.ST[option.steps - i + j, i]],
                    mode='markers',
                    text=[f'{option.ST[option.steps - i + j, i]:.2f}'],
                    textposition='top center',
                    marker=dict(size=10),
                    showlegend=False
                ))

    trin_fig.update_layout(
        title=dict(text="Trinomial Pricing Tree", x=0.5, xanchor='center', font=dict(size=24)),
        xaxis_title="Steps",
        yaxis_title="Price",
        xaxis=dict(tickmode='linear', dtick=1, range=[-0.5, steps + 0.5]),
        yaxis=dict(range=[min(option.ST.flatten()) * 0.9, max(option.ST.flatten()) * 1.1]),
        showlegend=False,
        height=800,
        width=800
    )

    # HTML and CSS for the text boxes
    call_box_html = make_text_box("Call Price", f"${call_price:.2f}", "#5DADE2")
    put_box_html = make_text_box("Put Price", f"${put_price:.2f}", "#FFA500")

    # Display the prices and tree plot
    col1, col2 = st.columns(2)
    with col1:
        st.write(call_box_html, unsafe_allow_html=True)
    with col2:
        st.write(put_box_html, unsafe_allow_html=True)

    st.plotly_chart(trin_fig, use_container_width=True)