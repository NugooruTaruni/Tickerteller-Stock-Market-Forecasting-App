import streamlit as st
from datetime import date
# import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import ta
import finnhub
# from plotly import graph_objs as go

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key="ct8bh4hr01qtkv5s163gct8bh4hr01qtkv5s1640") # ("YOUR_FINNHUB_KEY")

st.title('tickerteller')

# Stock picker drop-down
# stocks = ('GOOG', 'AAPL', 'MSFT', 'NVDA')
# selected_stock = st.selectbox('Select your ticker.', stocks)

# Ticker user-input
selected_stock = st.text_input('Enter your stock ticker (e.g., AAPL, GOOG, NVDA):', value='NVDA')
selected_stock = selected_stock.upper()

if not selected_stock:
    st.write("Enter a ticker symbol to get started.")
    st.stop()

def get_stock_details(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        return stock.info
    except Exception as e:
        return None

# Display stock details
stock_details = get_stock_details(selected_stock)

if stock_details:
    st.write(f"**Ticker**: {stock_details.get('symbol', 'N/A')}")
    st.write(f"**Name**: {stock_details.get('longName', 'N/A')}")
    st.write(f"**Market Cap**: ${stock_details.get('marketCap', 'N/A'):,}")
    # st.write(f"**Exchange**: {stock_details.get('exchange', 'N/A')}")
    st.write(f"**Industry**: {stock_details.get('industry', 'N/A')}")
    # st.write(f"**Website**: [Visit Website]({stock_details.get('website', 'N/A')})")
    st.write(f"**Website**: {stock_details.get('website', 'N/A')}")
else:
    st.error("Failed to fetch stock details. Please check the ticker or try again later.")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY, group_by='ticker')
    data.columns = [col[1] if col[0] == selected_stock else col[0] for col in data.columns]
    data.reset_index(inplace=True)
    # data = data[['Date', 'Open', 'Low', 'High', 'Close', 'Adj Close', 'Volume']]

    # Add moving averages
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    return data

# data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
# data_load_state.text('Loading complete!')

st.divider()

# Display news articles
st.subheader(f'Latest Financial Headlines for {selected_stock}')

def fetch_finnhub_news(ticker):
    return finnhub_client.company_news(ticker.upper(), _from=START, to=TODAY)

try:
    news_articles = fetch_finnhub_news(selected_stock)
    for idx, article in enumerate(news_articles[:5], start=1):  # Limit to 5 articles
        st.write(f"##### {idx}. {article['headline']}")
        st.write(article['summary'])
        st.write(f"[Read more...]({article['url']})")
        st.write("---")
except Exception as e:
    st.error("Failed to fetch news articles. Please check the stock ticker or try again later.")

# st.divider()

# Display raw data
st.subheader('Raw Data')
st.dataframe(data.tail(), use_container_width=True)

def plot_raw_data(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(data['Date'], data['Open'], label="Stock Open")
    ax.plot(data['Date'], data['Close'], label="Stock Close")
    ax.set_title('Time Series Data with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

plot_raw_data(data)

st.divider()


# Trend
st.subheader('Trend Analysis')

def plot_trend_data(data, show_sma50, show_sma200):
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(data['Date'], data['Open'], label="Stock Open")
    ax.plot(data['Date'], data['Close'], label="Stock Close")
    
    # Add moving averages based on user input
    if show_sma50:
        ax.plot(data['Date'], data['SMA50'], label="50-Day SMA", linestyle='--', alpha=0.7)
    if show_sma200:
        ax.plot(data['Date'], data['SMA200'], label="200-Day SMA", linestyle='--', alpha=0.7)
    
    ax.set_title('Time Series Data with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.text("A simple moving average smooths out volatility and makes it easier to view the price trend of a security. If the simple moving average points up, this means that the security's price is increasing. If it is pointing down, it means that the security's price is decreasing. The longer the time frame for the moving average, the smoother the simple moving average. A shorter-term moving average is more volatile, but its reading is closer to the source data.")

# Add checkboxes for moving averages
show_sma50 = st.checkbox('Show 50-Day SMA', value=True)
show_sma200 = st.checkbox('Show 200-Day SMA', value=True)
plot_trend_data(data, show_sma50, show_sma200)

st.caption(f"Simple Moving Average for {selected_stock}. The 200-day SMA is smoother, but the 50-day SMA is more reactive.")

st.divider()


# Momentum
st.subheader(f'Momentum Analysis')

# Function to plot RSI (Momentum Indicator)
def plot_rsi(data, window=14):
    rsi = ta.momentum.RSIIndicator(close=data['Close'], window=window).rsi()
    data['RSI'] = rsi
    
    # Streamlit Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, rsi, label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('RSI (Relative Strength Index)')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    st.pyplot(plt)

st.text(f"The relative strength index (RSI) is a momentum indicator that measures the speed and magnitude of a stock's recent price changes to detect overbought or oversold conditions in the price of that stock. It is displayed as an oscillator (a line graph) on a scale of zero to 100.")
plot_rsi(data)
st.caption(f"Relative Strength Index for {selected_stock}. A higher value indicates the stock is overbought and a lower value indicates that it's oversold.")

st.divider()

# Volume
st.subheader("Volume Analysis")

def plot_obv(data):
    obv = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
    data['OBV'] = obv
    
    # Streamlit Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, obv, label='OBV', color='blue')
    plt.title('On-Balance Volume (OBV)')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('OBV Value')
    st.pyplot(plt)

st.text("The on balance volume (OBV) indicator is a technical analysis tool used to detect the trading volume of an asset over time. OBV shows crowd sentiment that can predict a bullish or bearish outcome. It is believed that when volume increases sharply without a significant change in the stock's price, the price will eventually jump upward or fall downward.")
plot_obv(data)
st.caption(f"On Balance Volume for {selected_stock}.")

st.divider()

# Volatility
def plot_bollinger_bands(data, window=20, window_dev=3):

    # Initialize Bollinger Bands Indicator
    bands = ta.volatility.BollingerBands(close=data['Close'], window=window, window_dev=window_dev)

    # Add Bollinger Bands features to dataframe
    data['band_middle'] = bands.bollinger_mavg()
    data['band_high'] = bands.bollinger_hband()
    data['band_low'] = bands.bollinger_lband()

    # Add Bollinger Band high indicator
    data['band_high_ind'] = bands.bollinger_hband_indicator()

    # Add Bollinger Band low indicator
    data['band_low_ind'] = bands.bollinger_lband_indicator()

    # Add Width Size Bollinger Bands
    data['band_width'] = bands.bollinger_wband()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Close Price', color='blue')
    ax.plot(data['band_middle'], label='Bollinger MAVG (Middle Band)', color='green')
    ax.plot(data['band_high'], label='Bollinger High Band', color='red')
    ax.plot(data['band_low'], label='Bollinger Low Band', color='orange')
    ax.fill_between(data.index, data['band_low'], data['band_high'], color='grey', alpha=0.1)
    ax.set_title(f"Bollinger Bands")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

st.subheader(f'Volatility Analysis')
st.text(f"Bollinger Bands help gauge the volatility of stocks to determine if they are over- or undervalued. The center line is the stock price's 20-day simple moving average (SMA). The upper and lower bands are set at a certain number of standard deviations above and below the middle line. Stocks are usually interpreted to be overbought as their price nears the upper band and oversold as they approach the lower band.")
plot_bollinger_bands(data)
st.caption(f"Bollinger Bands for {selected_stock}. The band widens when the stock price is more volatile, and contracts when it's more stable.")

st.divider()

# Forecasting
st.subheader(":rocket: Forecasting")

option_years = [1, 2, 3, 4]
n_years = st.segmented_control("How many years ahead do you want to forecast?", option_years, default=1)
# n_years = st.slider('Years of prediction:', 1, 4)
period = int(n_years) * 365

# Prepare data for Prophet
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Train Prophet model
m = Prophet(weekly_seasonality=False)
m.fit(df_train)

# Create future dataframe and make predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Function to plot forecast
def plot_forecast():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_train['ds'], df_train['y'], label='Historical Data')  # Historical data
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')  # Predicted data
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='lightblue', alpha=0.3, label='Confidence Interval')  # Confidence interval
    ax.set_title('Stock Price Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Create forecast as a button-click
if st.button("Create Forecast"):
    # Show forecasted data
    st.subheader('Forecast Data')
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), use_container_width=True)

    # Convert forecast data to CSV
    forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)

    # Download button for forecast data
    st.download_button(
        label="Download Forecast Data as CSV",
        data=forecast_csv,
        file_name=f'{selected_stock}_forecast.csv',
        mime='text/csv',
        key="forecast_download_button"
    )
    st.divider()

    st.subheader('Forecast Plot')
    plot_forecast()
    st.divider()

    st.subheader('Forecast Components')
    st.write(f"Here we can observe the overall trend and the yearly seasonlity for {selected_stock}.")
    components = m.plot_components(forecast)
    st.pyplot(components)

# st.button("Create Forecast", on_click=create_forecast())
# TO-DO:
# change button functionality to on-click - DONE
# momentum indicator - DONE
# latest news - DONE