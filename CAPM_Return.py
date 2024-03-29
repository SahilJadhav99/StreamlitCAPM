# # importing libraries
# from datetime import date, datetime
# import streamlit as st

# import pandas as pd
# import yfinance as yf
# import pandas_datareader.data as web
# import datetime
# st.set_page_config(page_title="CAPM",
#                    page_icon="chart_with_upwards_trend",
#                    layout='wide')
# st.title("Capital A$$et Pricing Model")

# # To get input from user
# col1, col2 = st.columns([1, 1])
# with col1:
#     stocks_list = st.multiselect("Choose 4 Stocks", ('TSLA', 'AAPL', 'NFLX',
#                                  'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'), ['AAPL', 'AMZN', 'GOOGL'])
# with col2:
#     year = st.number_input("Number of Years", 1, 10)
# # Downloading data for SP500

# end = datetime.date.today()
# start = datetime.date(datetime.date.today().year-year,
#                       datetime.date.today().month, datetime.date.today().day)
# SP500 = web.DataReader(['sp500'], 'fred', start, end)
# stocks_df = pd.DataFrame()

# for stock in stocks_list:
#     data = yf.download(stock, period=f'{year}y')
#     stocks_df[f'{stock}'] = data['Close']

# stocks_df.reset_index(inplace=True)
# SP500.reset_index(inplace=True)
# SP500.columns = ['Date', 'sp500']
# stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
# stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
# stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
# stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')
# print(stocks_df)

# col1, col2 = st.columns([1, 1])
# with col1:
#     st.markdown('### Dataframe head')
#     st.dataframe(stocks_df.head(), use_container_width=True)
# with col2:
#     st.markdown('### Dataframe tail')
#     st.dataframe(stocks_df.tail(), use_container_width=True)

# importing libraries

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objs as go
from scipy import stats

# Set page configuration
st.set_page_config(page_title="CAPM & Risk Analysis",
                   page_icon="📉",
                   layout='wide')

# CSS styling with box shadow animation
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        color: #fff;
        padding-bottom: 20px;
        border-bottom: 2px solid #ccc;
        margin-bottom: 30px;
    }
    .section {
        padding: 20px;
        margin-bottom: 30px;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.5);
        transition: all 0.5s ease;
    }
    .section:hover {
        box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<p class="title">Capital Asset Pricing Model & Risk Analysis</p>',
            unsafe_allow_html=True)

# User input
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    stocks_list = st.multiselect("Choose Stocks", ('TSLA', 'NFLX',
                                                   'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'), ['AMZN', 'GOOGL'])
    frequency = st.radio("Data Frequency", ["Monthly", "Yearly"])
    year = st.number_input("Number of Years", 1, 10, 5)
    st.markdown('</div>', unsafe_allow_html=True)

# Customizable inputs
with col2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.5, 0.1)
    expected_return_market = st.slider(
        "Expected Market Return (%)", 0.0, 20.0, 8.0, 0.1)
    confidence_level = st.slider(
        "Confidence Level (%)", 0.0, 100.0, 95.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Downloading data for selected stocks
stocks_data = {}
for stock in stocks_list:
    if frequency == "Monthly":
        data = yf.download(stock, period=f'{year}y', interval='1mo')
    elif frequency == "Yearly":
        data = yf.download(stock, period=f'{year}y', interval='1y')
    stocks_data[stock] = data

# Plotting stock prices on a 3-month basis
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2>Stock Prices Over Time (3-Month Basis)</h2>',
            unsafe_allow_html=True)

# Create a figure
fig = go.Figure()

for stock, data in stocks_data.items():
    # Ensure that the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)
    # Resample data to 3-month frequency
    data_3m = data.resample('3M').mean()
    # Add trace to the figure
    fig.add_trace(go.Scatter(x=data_3m.index,
                             y=data_3m['Close'], mode='lines', name=stock))

# Update layout
fig.update_layout(title='Stock Prices Over Time (3-Month Basis)',
                  xaxis_title='Date',
                  yaxis_title='Stock Price')

# Show plot
st.plotly_chart(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Normalize stock prices
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2>Normalized Stock Prices</h2>', unsafe_allow_html=True)

# Create a DataFrame for normalized prices with the same index as the original data
normalized_prices = pd.DataFrame(index=data.index)

# Apply Min-Max normalization
for stock, data in stocks_data.items():
    # Ensure that the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)
    if not data.empty:  # Check if data is not empty
        # Normalize Close prices
        normalized_prices[stock] = (data['Close'] - data['Close'].min()) / \
            (data['Close'].max() - data['Close'].min())
    else:
        st.warning(
            f"No data available for {stock}. Please select another stock.")

# Plot normalized stock prices
if not normalized_prices.empty:  # Check if normalized_prices DataFrame is not empty
    fig_normalized = go.Figure()
    for stock in normalized_prices.columns:
        fig_normalized.add_trace(go.Scatter(
            x=data.index, y=normalized_prices[stock], mode='lines', name=stock))

    # Update layout
    fig_normalized.update_layout(title='Normalized Stock Prices',
                                 xaxis_title='Date',
                                 yaxis_title='Normalized Price')

    # Show plot
    st.plotly_chart(fig_normalized)
st.markdown('</div>', unsafe_allow_html=True)

# CAPM Formula
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2>Capital Asset Pricing Model (CAPM) - Formula</h2>',
            unsafe_allow_html=True)

st.markdown(
    """
    <div style="font-size: 18px; padding: 10px; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9; color: #333;">
    <p style="margin: 0;">$$ r_i = r_f + \\beta_i (r_m - r_f) $$</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<p style="font-size: 16px; color: #333;">Where:</p>',
            unsafe_allow_html=True)
st.markdown('- $r_i$ is the expected return of the asset',
            unsafe_allow_html=True)
st.markdown('- $r_f$ is the risk-free rate', unsafe_allow_html=True)
st.markdown('- $\\beta_i$ is the beta of the asset',
            unsafe_allow_html=True)
st.markdown('- $r_m$ is the expected return of the market',
            unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Calculate beta value
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2>Beta Value</h2>', unsafe_allow_html=True)

# Retrieve historical data for S&P 500 index
market_data = yf.download("^GSPC", period=f'{year}y', interval='1mo')

# Calculate returns for the market
market_returns = market_data['Close'].pct_change().dropna()

# Calculate beta for each stock
betas = {}
for stock, data in stocks_data.items():
    stock_returns = data['Close'].pct_change().dropna()
    beta, _, _, _, _ = stats.linregress(
        market_returns, stock_returns)
    betas[stock] = beta

# Display beta values
for stock, beta in betas.items():
    st.write(f"{stock}: {beta}")
st.markdown('</div>', unsafe_allow_html=True)

# Calculate expected returns using CAPM
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2>Expected Returns using CAPM</h2>', unsafe_allow_html=True)
expected_returns = {
    stock: risk_free_rate + beta * (expected_return_market / 100 - risk_free_rate) for stock, beta in betas.items()
}

expected_returns_df = pd.DataFrame(
    expected_returns.values(), index=expected_returns.keys(), columns=["Expected Return"])

# Display expected returns
st.write(expected_returns_df)
st.markdown('</div>', unsafe_allow_html=True)

# Additional Statistical Metrics
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2>Additional Statistical Metrics</h2>',
            unsafe_allow_html=True)
metrics_df = pd.DataFrame(index=stocks_list, columns=[
                          'Standard Deviation', 'Sharpe Ratio', "Jensen's Alpha"])

# Calculate metrics for each stock
for stock, data in stocks_data.items():
    stock_returns = data['Close'].pct_change().dropna()
    stock_std_dev = np.std(stock_returns)
    excess_return = stock_returns.mean() - risk_free_rate / 100
    sharpe_ratio = excess_return / stock_std_dev
    alpha, _, _, _, _ = stats.linregress(
        market_returns, stock_returns - risk_free_rate / 100)
    metrics_df.loc[stock] = [stock_std_dev, sharpe_ratio, alpha]

# Display metrics dataframe
st.write(metrics_df)

# Visualize metrics
fig_metrics = go.Figure()

fig_metrics.add_trace(go.Bar(x=metrics_df.index, y=metrics_df['Standard Deviation'],
                             name='Standard Deviation', marker_color='blue'))
fig_metrics.add_trace(go.Bar(x=metrics_df.index, y=metrics_df['Sharpe Ratio'],
                             name='Sharpe Ratio', marker_color='green'))
fig_metrics.add_trace(go.Bar(x=metrics_df.index, y=metrics_df["Jensen's Alpha"],
                             name="Jensen's Alpha", marker_color='orange'))

fig_metrics.update_layout(barmode='group',
                          title='Additional Statistical Metrics',
                          xaxis_title='Stocks',
                          yaxis_title='Value',
                          legend_title='Metrics')

st.plotly_chart(fig_metrics)
st.markdown('</div>', unsafe_allow_html=True)

# Risk Analysis
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2>Risk Analysis</h2>', unsafe_allow_html=True)

st.subheader("Value at Risk (VaR)")
st.write(f"Confidence Level: {confidence_level}%")
var_data = {'Stock': [], 'VaR': []}
for stock, data in stocks_data.items():
    returns = data['Close'].pct_change().dropna()
    var = norm.ppf(1 - confidence_level / 100,
                   returns.mean(), returns.std())
    var_data['Stock'].append(stock)
    var_data['VaR'].append(var)
var_df = pd.DataFrame(var_data)
st.table(var_df)

st.subheader("Conditional Value at Risk (CVaR)")
cvar_data = {'Stock': [], 'CVaR': []}
for stock, data in stocks_data.items():
    returns = data['Close'].pct_change().dropna()
    cvar = returns[returns <=
                   norm.ppf(1 - confidence_level / 100, returns.mean(), returns.std())].mean()
    cvar_data['Stock'].append(stock)
    cvar_data['CVaR'].append(cvar)
cvar_df = pd.DataFrame(cvar_data)
st.table(cvar_df)
st.markdown('</div>', unsafe_allow_html=True)
