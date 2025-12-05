import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
# from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="investIQ",page_icon='💰',
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("investIQ 📈")
st.header(" Advanced Stock Analysis & ML Prediction")

# Custom CSS
custom_css = """
<style>
    body {
        color: white;
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1e2130;
        border-right: 1px solid #3a3f5c;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #1e2130;
    }
    
    .stTextInput > div > div > input {
        color: white;
        background-color: #262730;
    }
    
    .stSelectbox > div > div > select {
        color: white;
        background-color: #262730;
    }
    
    .stDateInput > div > div > input {
        color: white;
        background-color: #262730;
    }
    
    .stMarkdown {
        color: white;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    .stNumberInput > div > div > input {
        color: white;
        background-color: #262730;
    }
    
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3a3f5c;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# List of popular stocks and indices
popular_tickers = [
    "^NSEI", "^BSESN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "LT.NS",
    "WIPRO.NS", "MARUTI.NS", "TATAMOTORS.NS", "ASIANPAINT.NS", "TITAN.NS",
    "AXISBANK.NS", "KOTAKBANK.NS",
    "RELIANCE.BO", "TCS.BO", "INFY.BO",
    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
]

ticker_display = {
    "^NSEI": "Nifty 50 Index (^NSEI)",
    "^BSESN": "Sensex Index (^BSESN)",
    "RELIANCE.NS": "Reliance Industries (RELIANCE.NS)",
    "TCS.NS": "Tata Consultancy Services (TCS.NS)",
    "INFY.NS": "Infosys (INFY.NS)",
    "HDFCBANK.NS": "HDFC Bank (HDFCBANK.NS)",
    "ICICIBANK.NS": "ICICI Bank (ICICIBANK.NS)",
    "AAPL": "Apple Inc. (AAPL)",
    "GOOGL": "Google (GOOGL)",
    "MSFT": "Microsoft (MSFT)",
}

# ==================== SIDEBAR ====================
st.sidebar.title("🎯 Configuration")
st.sidebar.markdown("---")

st.sidebar.header("📊 Stock Selection")

ticker = st.sidebar.selectbox(
    "Select Stock/Index:",
    options=popular_tickers,
    format_func=lambda x: ticker_display.get(x, x),
    index=0,
    help="Start with Nifty 50 or Sensex for best results"
)

st.sidebar.markdown("---")
st.sidebar.header("📅 Date Range")

today = datetime.now().date()
default_start = today - timedelta(days=730)  # 2 years for better ML training
default_end = today

start_date = st.sidebar.date_input(
    "Start Date:", 
    value=default_start,
    min_value=datetime(2000, 1, 1).date(),
    max_value=today,
    help="More historical data = better predictions"
)

end_date = st.sidebar.date_input(
    "End Date:", 
    value=default_end,
    min_value=start_date,
    max_value=today,
    help="Select the end date for historical data"
)

st.sidebar.markdown("---")
st.sidebar.header("🤖 ML Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Prediction Model:",
    [
        "Random Forest (Recommended)",
        "Gradient Boosting",
        "Support Vector Machine (SVR)",
        "Ridge Regression",
        "LSTM (Coming Soon)",
        "Ensemble (All Models)"
    ],
    help="Random Forest works best for stock prediction"
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Prediction Settings")

days_to_predict = st.sidebar.slider(
    "Days to Predict:",
    min_value=1,
    max_value=90,
    value=30,
    help="Number of days to forecast into the future"
)

test_size = st.sidebar.slider(
    "Test Data Size (%):",
    min_value=10,
    max_value=40,
    value=20,
    help="Percentage of data used for testing"
) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.header("📰 News Settings")

use_news = st.sidebar.checkbox("Show Market News", value=True, help="Display latest news for selected stock")

news_count = st.sidebar.slider(
    "Number of News Articles:",
    min_value=3,
    max_value=10,
    value=5,
    help="How many news articles to display"
)
st.sidebar.subheader("📋 Current Selection:")
st.sidebar.info(f"""
**Stock:** {ticker_display.get(ticker, ticker)}  
**Period:** {(end_date - start_date).days} days  
**Model:** {model_choice}  
**Forecast:** {days_to_predict} days
""")

# ==================== DATA LOADING ====================
@st.cache_data
def load_data(ticker, start, end):
    import time
    max_retries = 3
    retry_delay = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker, 
                start=start, 
                end=end, 
                progress=False,
                auto_adjust=False,
                prepost=False,
                threads=True,
                proxy=None
            )
            
            if data is None or len(data) == 0:
                last_error = f"No data returned (Attempt {attempt + 1}/{max_retries})"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None, last_error
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                last_error = f"Missing required columns. Found: {list(data.columns)}"
                return None, last_error
            
            data = data.dropna(how='all')
            
            if len(data) == 0:
                last_error = "All data rows were empty"
                return None, last_error
            
            return data, None
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None, last_error
    
    return None, last_error

# Feature Engineering Function
def create_features(data):
    """Create technical indicators and features for ML models"""
    df = data.copy()
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Volume features
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Rate of Change
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Lagged features
    for i in [1, 2, 3, 5, 7]:
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
    
    return df

@st.cache_data(ttl=600)
def fetch_stock_news(ticker, num_articles=5):
    """Fetch latest news for a stock using Google News RSS + Yahoo Finance as backup."""
    news_articles = []

    # --------- Build a search query for the ticker ----------
    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('^', '')

    if ticker == '^NSEI':
        search_query = 'Nifty 50 India stock market'
    elif ticker == '^BSESN':
        search_query = 'Sensex India stock market'
    else:
        # You can tweak this if you want more global news
        search_query = f'{clean_ticker} stock India market'

    # URL-encode the query safely
    encoded_query = quote_plus(search_query)

    # --------- Method 1: Google News RSS ----------
    try:
        rss_url = (
            f'https://news.google.com/rss/search?'
            f'q={encoded_query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en'
        )

        resp = requests.get(rss_url, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'xml')
            items = soup.find_all('item')[:num_articles]

            for item in items:
                title_tag = item.find('title')
                link_tag = item.find('link')
                date_tag = item.find('pubDate')
                source_tag = item.find('source')

                title = title_tag.text.strip() if title_tag else 'No title'
                link = link_tag.text.strip() if link_tag else '#'
                published = date_tag.text.strip() if date_tag else 'Recently'
                source = source_tag.text.strip() if source_tag else 'Google News'

                # Only add if we actually have a real title and link
                if title != 'No title' and link != '#':
                    news_articles.append({
                        'title': title,
                        'link': link,
                        'published': published,
                        'source': source,
                    })

                if len(news_articles) >= num_articles:
                    break
        else:
            st.warning(f"Google News RSS responded with status code {resp.status_code}")
    except Exception as e:
        st.warning(f"Could not fetch news from Google News RSS: {e}")

    # --------- Method 2: Yahoo Finance backup ----------
    # Only try backup if we still have fewer than requested
    if len(news_articles) < num_articles:
        try:
            stock = yf.Ticker(ticker)
            yahoo_news = getattr(stock, "news", []) or []

            for item in yahoo_news:
                # Some environments return weird structures; ensure it's a dict
                if not isinstance(item, dict):
                    continue

                title = item.get("title")
                link = item.get("link")
                if not title or not link:
                    # skip broken entries
                    continue

                ts = item.get("providerPublishTime")
                if ts is not None:
                    published = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                else:
                    published = "Recently"

                source = item.get("publisher", "Yahoo Finance")

                news_articles.append({
                    "title": title,
                    "link": link,
                    "published": published,
                    "source": source,
                })

                if len(news_articles) >= num_articles:
                    break

        except Exception as e:
            st.warning(f"Error while fetching Yahoo Finance news: {e}")

    return news_articles


def display_news(ticker, num_articles=5):
    """Display news in a nice format"""
    st.subheader(f"📰 Latest Market News - {ticker}")

    with st.spinner("Fetching latest news..."):
        news_articles = fetch_stock_news(ticker, num_articles)

    if not news_articles:
        st.info("No recent news found for this stock. Try another ticker or index like NIFTY / SENSEX.")
        return

    # Display news in cards
    for i, article in enumerate(news_articles, 1):
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**{i}. [{article['title']}]({article['link']})**")
                st.caption(f"📅 {article['published']} | 📰 {article['source']}")

            with col2:
                if st.button("Read", key=f"news_{i}"):
                    # Streamlit can't force-open a new tab, but we show the link clearly
                    st.write(f"Open this link in your browser: {article['link']}")

            st.markdown("---")



# Train ML Models
def train_models(X_train, X_test, y_train, y_test, model_name):
    """Train different ML models"""
    
    if model_name == "Random Forest (Recommended)":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "Support Vector Machine (SVR)":
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    elif model_name == "Ridge Regression":
        model = Ridge(alpha=1.0)
    elif model_name == "Ensemble (All Models)":
        # Train multiple models and average predictions
        models = {
            'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GB': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Ridge': Ridge(alpha=1.0)
        }
        predictions = []
        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            predictions.append(mdl.predict(X_test))
        
        y_pred = np.mean(predictions, axis=0)
        
        # For future predictions, we'll use the first model
        model = models['RF']
        return model, y_pred
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, y_pred

# ==================== MAIN CONTENT ====================

with st.spinner(f'Loading data for {ticker}...'):
    data, error_msg = load_data(ticker, start_date, end_date)

if data is None:
    st.error("❌ Unable to load data for the selected stock and date range.")
    if error_msg:
        st.code(f"Error: {error_msg}", language="text")
    st.stop()

if len(data) == 0:
    st.error("❌ No data available for the selected date range.")
    st.stop()

if 'Adj Close' not in data.columns and 'Close' in data.columns:
    data['Adj Close'] = data['Close']

st.success(f"✅ Successfully loaded {len(data)} days of data for {ticker}")

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Technical Analysis", "🤖 ML Predictions", "📉 Model Performance", "📰 Market News"])

with tab1:
    st.subheader("📊 Market Overview")
    
    # Basic statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
    with col2:
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
        st.metric("Total Change", f"₹{price_change:.2f}", f"{(price_change/data['Close'].iloc[0]*100):.2f}%")
    with col3:
        st.metric("Highest", f"₹{data['High'].max():.2f}")
    with col4:
        st.metric("Lowest", f"₹{data['Low'].min():.2f}")
    with col5:
        avg_volume = data['Volume'].mean()
        st.metric("Avg Volume", f"{avg_volume/1e6:.2f}M")
    
    # Price chart
    st.subheader("📉 Price Movement")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color='#00ff00', width=2)))
    fig.update_layout(
        title_text="Stock Price Over Time",
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Candlestick chart
    st.subheader("🕯️ Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(
        title_text="Candlestick Chart",
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume
    st.subheader("📊 Trading Volume")
    fig = go.Figure()
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=colors))
    fig.update_layout(
        title_text="Trading Volume",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📈 Technical Indicators")
    
    # Create features
    df_features = create_features(data)
    
    # Moving Averages
    st.subheader("📊 Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color='white', width=2)))
    fig.add_trace(go.Scatter(x=df_features.index, y=df_features['SMA_5'], name="SMA 5", line=dict(color='#FFD700')))
    fig.add_trace(go.Scatter(x=df_features.index, y=df_features['SMA_20'], name="SMA 20", line=dict(color='#FF6347')))
    fig.add_trace(go.Scatter(x=df_features.index, y=df_features['SMA_50'], name="SMA 50", line=dict(color='#4169E1')))
    fig.update_layout(
        title_text="Moving Averages",
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI
        st.subheader("📊 RSI (Relative Strength Index)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_features.index, y=df_features['RSI'], name="RSI", line=dict(color='#9370DB')))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MACD
        st.subheader("📊 MACD")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_features.index, y=df_features['MACD'], name="MACD", line=dict(color='#00CED1')))
        fig.add_trace(go.Scatter(x=df_features.index, y=df_features['MACD_Signal'], name="Signal", line=dict(color='#FF6347')))
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bollinger Bands
    st.subheader("📊 Bollinger Bands")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df_features.index, y=df_features['BB_Upper'], name="Upper Band", line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df_features.index, y=df_features['BB_Middle'], name="Middle Band", line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=df_features.index, y=df_features['BB_Lower'], name="Lower Band", line=dict(color='green', dash='dash')))
    fig.update_layout(
        title_text="Bollinger Bands",
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🤖 Machine Learning Predictions")
    
    # Prepare data for ML
    df_ml = create_features(data)
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 100:
        st.error("Not enough data for ML prediction. Please select a longer date range.")
        st.stop()
    
    # Select features for prediction
    feature_cols = [
        'Open', 'High', 'Low', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26', 'MACD', 'RSI',
        'BB_Width', 'Volatility', 'Volume_Ratio',
        'Momentum', 'ROC', 'ATR'
    ]
    
    # Add lagged features
    for i in [1, 2, 3, 5]:
        if f'Close_Lag_{i}' in df_ml.columns:
            feature_cols.append(f'Close_Lag_{i}')
    
    # Ensure all feature columns exist
    feature_cols = [col for col in feature_cols if col in df_ml.columns]
    
    X = df_ml[feature_cols]
    y = df_ml['Close']
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Split data
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Train model
    with st.spinner(f'Training {model_choice} model...'):
        model, y_pred_scaled = train_models(X_train, X_test, y_train, y_test, model_choice)
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100
    
    # Display metrics
    st.subheader("📊 Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("RMSE", f"₹{rmse:.2f}")
    with col2:
        st.metric("MAE", f"₹{mae:.2f}")
    with col3:
        st.metric("R² Score", f"{r2:.4f}")
    with col4:
        st.metric("MAPE", f"{mape:.2f}%")
    with col5:
        accuracy = max(0, (1 - mape/100) * 100)
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Plot predictions vs actual
    st.subheader("📈 Actual vs Predicted Prices")
    test_dates = df_ml.index[split_idx:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, mode='lines', name='Actual', line=dict(color='cyan', width=2)))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='Predicted', line=dict(color='red', width=2, dash='dash')))
    fig.update_layout(
        title_text=f"Actual vs Predicted - {model_choice}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Future predictions
    st.subheader(f"🔮 Future Price Forecast ({days_to_predict} Days)")
    
    # Prepare future predictions
    last_data = df_ml[feature_cols].iloc[-1:].copy()
    future_predictions = []
    
    for day in range(days_to_predict):
        # Scale the features
        last_scaled = scaler_X.transform(last_data)
        
        # Predict
        next_price_scaled = model.predict(last_scaled)[0]
        next_price = scaler_y.inverse_transform([[next_price_scaled]])[0][0]
        future_predictions.append(next_price)
        
        # Update last_data for next iteration (simplified - use last prediction)
        # In reality, you'd update all features based on the new price
        last_data = last_data.copy()
        if 'Close_Lag_1' in feature_cols:
            last_data['Close_Lag_1'] = next_price
    
    # Create future dates
    last_date = df_ml.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict, freq='D')
    
    # Plot future predictions
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data.index[-90:], 
        y=data['Close'].iloc[-90:], 
        name='Historical', 
        line=dict(color='lightblue', width=2)
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_predictions, 
        name='Forecast', 
        line=dict(color='red', width=2, dash='dash'),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title_text=f"{days_to_predict}-Day Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction table
    with st.expander("📋 View Detailed Forecast"):
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price (₹)': [f"₹{p:.2f}" for p in future_predictions],
            'Change from Today': [f"{((p - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%" 
                                  for p in future_predictions]
        })
        pred_df['Date'] = pred_df['Date'].dt.date
        st.dataframe(pred_df, use_container_width=True)
    
    # Prediction summary
    st.subheader("📊 Forecast Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Price (30 days)", 
            f"₹{future_predictions[min(29, len(future_predictions)-1)]:.2f}",
            f"{((future_predictions[min(29, len(future_predictions)-1)] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%"
        )
    
    with col2:
        max_pred = max(future_predictions)
        st.metric("Highest Forecast", f"₹{max_pred:.2f}")
    
    with col3:
        min_pred = min(future_predictions)
        st.metric("Lowest Forecast", f"₹{min_pred:.2f}")

with tab4:
    st.subheader("📉 Model Performance Analysis")

    # Residuals (Actual - Predicted)
    residuals = y_test_actual - y_pred

    col1, col2 = st.columns(2)

    # ---------------- Residuals Distribution ----------------
    with col1:
        st.subheader("📊 Prediction Errors Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=50, edgecolor='black')
        ax.axvline(x=0, linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error (₹)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Errors')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # ---------------- Residuals vs Predicted ----------------
    with col2:
        st.subheader("📊 Residuals vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Price (₹)')
        ax.set_ylabel('Residuals (₹)')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # ---------------- Feature Importance ----------------
    if model_choice in ["Random Forest (Recommended)", "Gradient Boosting"]:
        st.subheader("🎯 Feature Importance")

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h'
            ))
            fig.update_layout(
                title_text="Top 15 Most Important Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- Model Comparison ----------------
    st.subheader("📊 Quick Model Comparison")

    if st.button("Compare All Models"):
        with st.spinner("Training all models for comparison..."):
            models_to_compare = {
                "Random Forest": RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=42
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=50, max_depth=5, random_state=42
                ),
                "Ridge": Ridge(alpha=1.0),
            }

            comparison_results = []

            for name, mdl in models_to_compare.items():
                mdl.fit(X_train, y_train)
                y_pred_temp = mdl.predict(X_test)
                y_pred_temp_actual = scaler_y.inverse_transform(
                    y_pred_temp.reshape(-1, 1)
                ).ravel()

                rmse_temp = np.sqrt(mean_squared_error(y_test_actual, y_pred_temp_actual))
                r2_temp = r2_score(y_test_actual, y_pred_temp_actual)
                mape_temp = mean_absolute_percentage_error(
                    y_test_actual, y_pred_temp_actual
                ) * 100

                comparison_results.append({
                    'Model': name,
                    'RMSE (₹)': f"{rmse_temp:.2f}",
                    'R² Score': f"{r2_temp:.4f}",
                    'MAPE (%)': f"{mape_temp:.2f}"
                })

        comparison_df = pd.DataFrame(comparison_results)
        st.dataframe(comparison_df, use_container_width=True)
        st.info("💡 Lower RMSE and MAPE, higher R² indicate better model performance")

with tab5:
    if use_news:
        display_news(ticker, news_count)
    else:
        st.info("Enable 'Show Market News' in the sidebar to view latest news.")
           

# Disclaimer
st.markdown("---")
st.warning("""
⚠️ Important Disclaimer:

These predictions are generated by machine learning models and should NOT be used as the sole basis for investment decisions
Past performance does not guarantee future results
Stock markets are influenced by many unpredictable factors
Always conduct your own research and consult with financial advisors
Consider your risk tolerance and investment goals
The models' accuracy varies with market conditions
""")

# Footer
# --- Footer ---
st.markdown("---")

linkedin_url = "https://www.linkedin.com/in/abhaysingh212003/"
github_url = "https://github.com/03abhay"

footer_html = f"""
<style>
.footer {{
    text-align: center;
    color: gray;
    font-size: 0.9rem;
    margin-top: 20px;
}}
.footer a {{
    color: #58a6ff;
    text-decoration: none;
}}
.footer a:hover {{
    text-decoration: underline;
}}
.github-btn {{
    display: inline-block;
    padding: 6px 14px;
    margin-left: 8px;
    border-radius: 999px;
    border: 1px solid #58a6ff;
    color: #58a6ff;
    font-size: 0.85rem;
    text-decoration: none;
}}
.github-btn:hover {{
    background-color: #58a6ff22;
}}
</style>

<div class="footer">
    <p>Built with ❤️ using Streamlit & Advanced ML &nbsp;|&nbsp; Data powered by Yahoo Finance</p>
    <p>
        <a href="{linkedin_url}" target="_blank" class="LinkedIn-btn">Connect on LinkedIn</a>
        <a href="{github_url}" target="_blank" class="github-btn">GitHub Profile</a>
    </p>
    <p>© 2025 InvestIQ by Abhay Singh. All rights reserved.</p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
