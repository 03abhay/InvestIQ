import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from textblob import TextBlob
from plotly.subplots import make_subplots
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# **PAGE CONFIGURATION SECTION**
st.set_page_config(page_title="InvestIQ", layout="wide", page_icon='icon.png')

# **LOGO AND TITLE SECTION**
c1, c2, c3, = st.columns(3)
with c2:
    st.image("icon.png", width=150)

col1, col2, col3 = st.columns(3)
with col2:
    st.title("Stocks Analysis")

# **SIDEBAR STOCK SELECTION**
st.sidebar.header("Make Inputs for Analysis")

# **INDIAN STOCKS LIST** 
indian_stocks = [
    "ADANITOTAL.NS", "ADANIGREEN.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ADANITRANS.NS", 
    "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS", "AWL.NS", 
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BANDHANBNK.NS", "BANKBARODA.NS", 
    "BHARTIARTL.NS", "BIOCON.NS", "BPCL.NS", "BOSCHLTD.NS", "BRITANNIA.NS", 
    "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "CONCOR.NS", 
    "DABUR.NS", "DIVISLAB.NS", "DRREDDY.NS", 
    "EICHERMOT.NS", 
    "GAIL.NS", "GRASIM.NS", "GODREJCP.NS", "GODREJPROP.NS", 
    "HAVELLS.NS", "HDFC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "HCLTECH.NS", 
    "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDFCFIRSTB.NS", "INDIGO.NS", "INDUSINDBK.NS", "INFY.NS", "IRCTC.NS", "IOC.NS", 
    "ITC.NS", 
    "JUBLFOOD.NS", "JSWSTEEL.NS", 
    "KOTAKBANK.NS", 
    "LT.NS", "LUPIN.NS", 
    "M&M.NS", "MANAPPURAM.NS", "MARUTI.NS", "MAXHEALTH.NS", "MCDOWELL-N.NS", "MRF.NS", "MUTHOOTFIN.NS", 
    "NESTLEIND.NS", "NHPC.NS", "NYKAA.NS", "NTPC.NS", 
    "ONGC.NS", 
    "PAGEIND.NS", "PEL.NS", "PNB.NS", "PETRONET.NS", "PIDILITIND.NS", "POWERGRID.NS", "PVR.NS", 
    "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", 
    "SBIN.NS", "SBICARD.NS", "SBILIFE.NS", "SHREECEM.NS", "SIEMENS.NS", "SUNPHARMA.NS", 
    "TATACHEM.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TATAMOTORS.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", 
    "TVSMOTOR.NS", 
    "UPL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", 
    "WIPRO.NS", 
    "ZEEL.NS", "ZOMATO.NS"
]

selected_ticker = st.sidebar.selectbox(
    "Select a Stock Ticker from the List",
    ["Custom"] + indian_stocks,
)

# **TICKER INPUT LOGIC**
if selected_ticker == "Custom":
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", "")
else:
    ticker = selected_ticker

# **CHECK TICKER INPUT**
if not ticker:
    st.warning("Please select a ticker from the list or enter a custom ticker.")
    st.stop()

# Time Period Selection with Unique Keys
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ("Custom", "1 Week", "1 Month", "1 Year", "5 Years"),
    key="time_period_select"  
)

end_date = date.today()

if time_period == "Custom":
    # Add unique keys to date inputs
    start_date = st.sidebar.date_input(
        "Start Date", 
        end_date - timedelta(days=365), 
        key="start_date_input"  # Unique key for start date
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        end_date, 
        key="end_date_input"  # Unique key for end date
    )
elif time_period == "1 Week":
    start_date = end_date - timedelta(days=7)
elif time_period == "1 Month":
    start_date = end_date - timedelta(days=30)
elif time_period == "1 Year":
    start_date = end_date - timedelta(days=365)
else:  # 5 Years
    start_date = end_date - timedelta(days=365*5)
# **DATA LOADING FUNCTION**
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# **LOAD STOCK DATA**
data = load_data(ticker, start_date, end_date)

if data.empty:
    st.warning("No data available for the selected stock and date range.")
    st.stop()

# **VISUALIZATIONS** 
# Raw Data Display
st.subheader("Raw Data")
st.write(data)



# Stock Price Chart
st.subheader("Stock Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close"))
fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="Open"))
fig.layout.update(title_text="Stock Price Over Time", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# **NEWS SENTIMENT ANALYSIS FUNCTIONS**
def fetch_news(stock_ticker):
    
    clean_ticker = stock_ticker.split('.')[0]
    
   

    API_KEY = "cc00ed756c0f4e8d92aa780e12d708e8"
    
    url = f"https://newsapi.org/v2/everything?q={clean_ticker}&sortBy=publishedAt&language=en&apiKey={API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        news_data = response.json()
        articles = news_data.get("articles", [])
        
        # Filter out articles without titles
        filtered_articles = [article for article in articles if article.get("title")]
        
        return filtered_articles
    
    except requests.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []

# **SENTIMENT ANALYSIS FUNCTION**
def analyze_sentiment(headlines):
    sentiments = []
    for headline in headlines:
        analysis = TextBlob(headline)
        sentiments.append(analysis.sentiment.polarity)
    return sentiments

# **NEWS SENTIMENT DISPLAY FUNCTION**
def display_news_sentiment(ticker):
    st.subheader("News Sentiment Analysis")
    
    try:
        # Fetch news articles
        news_articles = fetch_news(ticker)
        
        if not news_articles:
            st.warning("No news articles found for the selected stock.")
            return
        
        # Prepare data for visualization with clickable links
        sentiment_df = pd.DataFrame({
            "Headline": [f"[{article['title']}]({article['url']})" for article in news_articles],
            "Sentiment": analyze_sentiment([article["title"] for article in news_articles]),
            "Source": [article.get("source", {}).get("name", "Unknown Source") for article in news_articles]
        })
        
        # Display headlines with sentiment scores and sources
        st.write("Recent News Headlines with Sentiment Scores")
        st.markdown(sentiment_df.to_markdown(index=False), unsafe_allow_html=True)
        
        # Visualize sentiment distribution
        fig = px.histogram(sentiment_df, x="Sentiment", nbins=20, title="Sentiment Distribution")
        st.plotly_chart(fig)
        
        # Overall sentiment
        avg_sentiment = sentiment_df["Sentiment"].mean()
        if avg_sentiment > 0.1:
            overall_sentiment = "Positive"
        elif avg_sentiment < -0.1:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        st.write(f"Overall Sentiment: **{overall_sentiment}**")
        st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
    
    except Exception as e:
        st.error(f"An error occurred during news sentiment analysis: {e}")

# **CALL NEWS SENTIMENT ANALYSIS**
display_news_sentiment(ticker)






# **VISUALIZATIONS**



data['Returns'] = data['Close'].pct_change()

# Candlestick Chart
st.subheader("Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
st.plotly_chart(fig)

# Volume Chart
volume_data = data['Volume']
volume_dates = data.index

fig = go.Figure(data=[go.Bar(x=volume_dates, y=volume_data)])
fig.update_layout(title='Trading Volume', xaxis_title='Date', yaxis_title='Volume')
st.plotly_chart(fig)

# Returns Distribution
st.subheader("Returns Distribution")
fig = px.histogram(data, x='Returns', nbins=50, title='Daily Returns Distribution')
st.plotly_chart(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", title='Feature Correlation')
st.plotly_chart(fig)

# Simple Moving Averages
data['SMA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
data['SMA50'] = data['Close'].rolling(window=50, min_periods=1).mean()

st.subheader("Moving Averages")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Close Price"))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], mode='lines', name="20-Day SMA"))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines', name="50-Day SMA"))
fig.update_layout(title='Stock Price with Moving Averages')
st.plotly_chart(fig)
st.subheader("Understanding Simple Moving Averages (SMA)")
st.write( "What are Simple Moving Averages?\n Simple Moving Averages (SMA) are a fundamental technical analysis indicator that helps smooth out price data by creating a constantly updated average price")

st.write("How SMAs Work \n- **20-Day SMA**: Calculates the average closing price over the last 20 trading days \n- **50-Day SMA**: Calculates the average closing price over the last 50 trading days\n")


# **Price Prediction Model**
st.subheader("")
st.subheader("\n \n Price Prediction (Next 30 Days)")

# Prepare the data for prediction
data['Days'] = (data.index - data.index[0]).days  # Convert date index to number of days
data['Prediction'] = data['Close'].shift(-1)

# Drop rows with NaN values due to shift operation
data = data.dropna()

# Features and target variables
X = data[['Days']]
y = data['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate future days for predictions (next 30 days)
future_days = np.arange(len(data), len(data) + 30)
future_df = pd.DataFrame({'Days': future_days})

# Predict the prices for the next 30 days
future_predictions = model.predict(future_df)

# Visualization of actual vs predicted prices
# Create future dates explicitly
future_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Actual Close"))
fig.add_trace(go.Scatter(x=future_dates, 
                         y=future_predictions, 
                         mode='lines',
                         name="Predicted Close", 
                         line=dict(color='red', dash='dot')))
fig.update_layout(title='Stock Price Prediction')
st.plotly_chart(fig)

# Model Performance Metrics on Test Data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²) Score: {r2:.2f}")

# Option to use Ridge regression for better regularization
ridge_model = Ridge(alpha=45.0)  
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(future_df)

# Visualization of Ridge regression predictions
fig_ridge = go.Figure()
fig_ridge.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Actual Close"))
fig_ridge.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=31, freq='D')[1:], 
                               y=ridge_predictions, 
                               name="Ridge Predicted Close", 
                               line=dict(color='green', dash='dot')))
fig_ridge.update_layout(title='Stock Price Prediction with Ridge Regression')
st.plotly_chart(fig_ridge)

# Performance Metrics for Ridge Regression
y_ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)
ridge_r2 = r2_score(y_test, y_ridge_pred)



st.subheader("\nModel Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Squared Error", f"{mse:.4f}")
with col2:
    st.metric("R-squared Score", f"{r2:.4f}")

st.write("Note: This is a simple linear regression model and should not be used for actual trading decisions.")

# **FOOTER WITH PERSONAL LINKS**
def add_comprehensive_footer():
    st.markdown("""
    <style>
    .comprehensive-footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 20px;
        border-top: 1px solid #e0e0e0;
        color: #666;
        font-size: 0.9em;
    }
    .comprehensive-footer a {
        color: #007bff;
        text-decoration: none;
        margin: 0 10px;
        transition: color 0.3s ease;
    }
    .comprehensive-footer a:hover {
        color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='comprehensive-footer'>
    © 2024 InvestIQ. All Rights Reserved | 
    Developed by  Abhay Singh  | 
    <a href='https://github.com/03abhay' target='_blank'>GitHub</a> | 
    <a href='https://www.linkedin.com/in/abhaysingh212003/' target='_blank'>LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# Add this at the end of your Streamlit script
add_comprehensive_footer()
