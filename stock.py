import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import time
import pytz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

# Set up API key and model
GOOGLE_API_KEY = "AIzaSyBf3Y7EVko2N7MKvE4RXWUxxecL_qNdruU"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Streamlit config
st.set_page_config(layout="wide")
st.title("Advanced Indian Stock Market Analysis and Prediction App")

# Sidebar
st.sidebar.title("Market Recommendations")

# Function to get market recommendations
def get_market_recommendations():
    prompt = """
    Based on the current Indian stock market conditions, provide a list of 5 stocks to buy and 5 stocks to sell.
    For each stock, provide a brief reason for the recommendation.
    Format the response as follows:

    Stocks to Buy:
    1. SYMBOL1: Reason
    2. SYMBOL2: Reason
    3. SYMBOL3: Reason
    4. SYMBOL4: Reason
    5. SYMBOL5: Reason

    Stocks to Sell:
    1. SYMBOL6: Reason
    2. SYMBOL7: Reason
    3. SYMBOL8: Reason
    4. SYMBOL9: Reason
    5. SYMBOL10: Reason
    """
    response = model.generate_content(prompt)
    return response.text

# Display market recommendations
market_recommendations = get_market_recommendations()
st.sidebar.write(market_recommendations)

# User input
user_input = st.text_input("Enter NSE stock symbol (e.g., RELIANCE, TCS):", "RELIANCE")
stock_symbol = user_input if user_input.endswith('.NS') else f"{user_input}.NS"

# Data functions
@st.cache_data(ttl=60)
def get_real_time_stock_data(symbol):
    stock = yf.Ticker(symbol)
    return stock.history(period="1d", interval="1m")

@st.cache_data(ttl=3600)
def get_historical_stock_data(symbol):
    stock = yf.Ticker(symbol)
    return stock.history(period="5y")

# Session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = get_real_time_stock_data(stock_symbol)
    st.session_state.historical_data = get_historical_stock_data(stock_symbol)

# Real-time metrics
def update_real_time_metrics():
    current_data = get_real_time_stock_data(stock_symbol)
    current_price = current_data['Close'].iloc[-1]
    open_price = current_data['Open'].iloc[0]
    high_price = current_data['High'].max()
    low_price = current_data['Low'].min()
    volume = current_data['Volume'].sum()
    return current_price, open_price, high_price, low_price, volume

metrics_placeholder = st.empty()

def display_real_time_metrics(current_price, open_price, high_price, low_price, volume):
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"₹{current_price:.2f}", f"{((current_price - open_price) / open_price) * 100:.2f}%")
        col2.metric("Open Price", f"₹{open_price:.2f}")
        col3.metric("Volume", f"{volume:,}")

        col4, col5 = st.columns(2)
        col4.metric("Day High", f"₹{high_price:.2f}")
        col5.metric("Day Low", f"₹{low_price:.2f}")

current_price, open_price, high_price, low_price, volume = update_real_time_metrics()
display_real_time_metrics(current_price, open_price, high_price, low_price, volume)
st.write(f"Real-time data from: {st.session_state.stock_data.index[0]} to {st.session_state.stock_data.index[-1]}")

# News headlines
@st.cache_data(ttl=3600)
def get_news_headlines(symbol):
    company_name = symbol.replace('.NS', '')
    url = f"https://news.google.com/rss/search?q={company_name}+stock+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features='xml')
    items = soup.findAll('item')
    return [item.title.text for item in items[:5]]

headlines = get_news_headlines(stock_symbol)

st.subheader("Recent News Headlines")
for headline in headlines:
    st.write(f"• {headline}")

# Sentiment Analysis
def analyze_sentiment(headlines):
    prompt = f"Analyze the sentiment of these headlines for {stock_symbol} stock. Provide a brief summary and rate the overall sentiment as positive, neutral, or negative:\n\n" + "\n".join(headlines)
    response = model.generate_content(prompt)
    return response.text

sentiment_analysis = analyze_sentiment(headlines)
st.subheader("Sentiment Analysis")
st.write(sentiment_analysis)

# Helper indicators
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd - signal

# Stock prediction
def predict_stock_price(symbol):
    data = get_historical_stock_data(symbol)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    data['Volatility'] = data['Close'].rolling(window=20).std()

    features = ['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'Volatility']
    data_aligned = data.dropna()
    X = data_aligned[features]
    y = data_aligned['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    last_data_point = X.iloc[-1].values.reshape(1, -1)
    rf_prediction = rf_model.predict(last_data_point)[0]

    arima_model = ARIMA(y, order=(1, 1, 1))
    arima_results = arima_model.fit()
    arima_forecast = arima_results.forecast(steps=1)
    arima_prediction = arima_forecast.iloc[0] if isinstance(arima_forecast, pd.Series) else arima_forecast[0]

    ensemble_prediction = (rf_prediction + arima_prediction) / 2
    return data['Close'].iloc[-1], ensemble_prediction, data.index[-1]

last_price, predicted_price, last_refreshed = predict_stock_price(stock_symbol)

st.subheader("Advanced Stock Prediction")
st.write(f"Last closing price: ₹{last_price:.2f}")
st.write(f"Predicted price (Ensemble of Random Forest and ARIMA): ₹{predicted_price:.2f}")
st.write(f"Data last refreshed: {last_refreshed}")

# Multi-timeframe prediction
def predict_multiple_timeframes(symbol):
    data = get_historical_stock_data(symbol)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    data['Volatility'] = data['Close'].rolling(window=20).std()

    features = ['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'Volatility']
    data_aligned = data.dropna()
    X = data_aligned[features]
    y = data_aligned['Close']

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    arima_model = ARIMA(y, order=(1, 1, 1))
    arima_results = arima_model.fit()
    last_data_point = X.iloc[-1].values.reshape(1, -1)

    rf_intraday = rf_model.predict(last_data_point)[0]
    arima_intraday = arima_results.forecast(steps=1).iloc[0]
    intraday_prediction = (rf_intraday + arima_intraday) / 2

    rf_short_term = rf_model.predict(last_data_point)[0] * (1 + np.random.normal(0, 0.02))
    arima_short_term = arima_results.forecast(steps=5).iloc[-1]
    short_term_prediction = (rf_short_term + arima_short_term) / 2

    rf_long_term = rf_model.predict(last_data_point)[0] * (1 + np.random.normal(0, 0.05))
    arima_long_term = arima_results.forecast(steps=20).iloc[-1]
    long_term_prediction = (rf_long_term + arima_long_term) / 2

    next_date = data.index[-1] + timedelta(days=1)
    return next_date, intraday_prediction, short_term_prediction, long_term_prediction

next_date, intraday_pred, short_term_pred, long_term_pred = predict_multiple_timeframes(stock_symbol)

st.subheader(f"Predictions for {next_date.date()}")
st.write(f"Intraday (Next day): ₹{intraday_pred:.2f}")
st.write(f"Short term (1 week): ₹{short_term_pred:.2f}")
st.write(f"Long term (1 month): ₹{long_term_pred:.2f}")

# Buy/Sell prediction
def predict_buy_sell(symbol, last_price, predicted_price, sentiment_analysis):
    prompt = f"""
    Based on the following information for {symbol} stock:
    1. Last closing price: ₹{last_price:.2f}
    2. Predicted price (Ensemble model): ₹{predicted_price:.2f}
    3. Sentiment analysis: {sentiment_analysis}

    Please provide:
    Recommendation: [BUY/SELL/HOLD]
    Potential price change: [X.XX%]
    Explanation: [Your explanation here]
    Risks and factors: [List potential risks and factors]
    """
    response = model.generate_content(prompt)
    return response.text

buy_sell_prediction = predict_buy_sell(stock_symbol, last_price, predicted_price, sentiment_analysis)
st.subheader("Buy/Sell Prediction")
st.write(buy_sell_prediction)

# Historical performance
hist_data = st.session_state.historical_data
st.subheader("Historical Performance")
st.write(f"1-Year Return:  {((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-252]) - 1) * 100:.2f}%")
st.write(f"1-Year High: ₹{hist_data['High'][-252:].max():.2f}")
st.write(f"1-Year Low: ₹{hist_data['Low'][-252:].min():.2f}")
st.write(f"Historical data from: {hist_data.index[0]} to {hist_data.index[-1]}")

# Technical analysis
st.subheader("Technical Analysis")
ta_data = hist_data.copy()
ta_data['MA50'] = ta_data['Close'].rolling(window=50).mean()
ta_data['MA200'] = ta_data['Close'].rolling(window=200).mean()
ta_data['RSI'] = calculate_rsi(ta_data['Close'])
ta_data['MACD'] = calculate_macd(ta_data['Close'])

col1, col2 = st.columns(2)
with col1:
    st.write("RSI (14-day):", f"{ta_data['RSI'].iloc[-1]:.2f}")
    if ta_data['RSI'].iloc[-1] > 70:
        st.write("RSI indicates overbought conditions")
    elif ta_data['RSI'].iloc[-1] < 30:
        st.write("RSI indicates oversold conditions")
    else:
        st.write("RSI is in neutral territory")
with col2:
    st.write("MACD:", f"{ta_data['MACD'].iloc[-1]:.2f}")
    if ta_data['MACD'].iloc[-1] > 0:
        st.write("MACD is above the signal line, indicating bullish momentum")
    else:
        st.write("MACD is below the signal line, indicating bearish momentum")

# Generate text report
def generate_text_report():
    report = f"""
Stock Analysis Report for {stock_symbol}
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Current Metrics:
Current Price: ₹{current_price:.2f}
Open Price: ₹{open_price:.2f}
Day High: ₹{high_price:.2f}
Day Low: ₹{low_price:.2f}
Volume: {volume:,}

Stock Predictions:
Last Closing Price: ₹{last_price:.2f}
Ensemble Prediction: ₹{predicted_price:.2f}
Intraday ({next_date.date()}): ₹{intraday_pred:.2f}
Short Term (1 week): ₹{short_term_pred:.2f}
Long Term (1 month): ₹{long_term_pred:.2f}

Historical Performance:
1-Year Return: {((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-252]) - 1) * 100:.2f}%
1-Year High: ₹{hist_data['High'][-252:].max():.2f}
1-Year Low: ₹{hist_data['Low'][-252:].min():.2f}

Technical Indicators:
RSI (14-day): {ta_data['RSI'].iloc[-1]:.2f}
MACD: {ta_data['MACD'].iloc[-1]:.2f}

Sentiment Analysis:
{sentiment_analysis}

Buy/Sell Prediction:
{buy_sell_prediction}

Disclaimer: This report is for educational purposes only. Do not use it for actual trading decisions.
    """
    return report

if st.button("Download Complete Report (TXT)"):
    report = generate_text_report()
    st.download_button(
        label="Click here to download the text report",
        data=report,
        file_name="stock_analysis_report.txt",
        mime="text/plain"
    )

# Disclaimer
st.sidebar.warning("Disclaimer: This app is for educational purposes only. Do not use it for actual trading decisions.")
