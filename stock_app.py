# Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
import requests
import ta  


api_key = "1a9444f6065ad0fce6a13525442c3c8727c92868"  
news_api_key = "aebf914e21f7429d8416bf02c269b305"  

default_start_date = '2023-10-16'
default_end_date = '2024-10-16'


st.title('Enhanced Stock Price Prediction and Analysis')


ticker_symbol = st.text_input("Enter the stock ticker symbol", value="AAPL").upper()
url = f"https://api.tiingo.com/tiingo/daily/{ticker_symbol}/prices"


params = {
    'startDate': default_start_date,
    'endDate': default_end_date,
    'token': api_key
}
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Adding Technical Indicators (SMA, RSI, Bollinger Bands)
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['close'])
    df['Upper Band'] = ta.volatility.bollinger_hband(df['close'])
    df['Lower Band'] = ta.volatility.bollinger_lband(df['close'])

    st.subheader(f"{ticker_symbol} Stock Price and Technical Indicators")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], line=dict(color='green', width=1), name="Upper Bollinger Band"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], line=dict(color='red', width=1), name="Lower Bollinger Band"))

    fig.update_layout(title=f"{ticker_symbol} Stock Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Data Preprocessing for LSTM Model
    df2 = df[['close']].reset_index()
    scaler = MinMaxScaler()
    df2[['close']] = scaler.fit_transform(df2[['close']])

    # Train-test split
    train_size = int(len(df2) * 0.8)
    train, test = df2[:train_size], df2[train_size:]

    # Function to create dataset for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 25
    X_train, y_train = create_dataset(train[['close']].values, time_step)
    X_test, y_test = create_dataset(test[['close']].values, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Predictions and model evaluation
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_predict)
    test_mae = mean_absolute_error(y_test, test_predict)

    # Predicting the next day's closing price
    def predict_next_day(model, last_data, time_step):
        last_data = last_data[-time_step:].reshape(1, time_step, 1)
        prediction = model.predict(last_data)
        return scaler.inverse_transform(prediction)

    last_closing_price = df2['close'].values[-time_step:]
    predicted_tomorrow = predict_next_day(model, last_closing_price, time_step)
    st.markdown(f"<h2 style='color: blue;'>Predicted Closing Price for Tomorrow: <strong>${predicted_tomorrow[0, 0]:.4f}</strong></h2>", unsafe_allow_html=True)
else:
    st.error("Error fetching stock data.")

# News API Integration
st.subheader(f"Latest News for {ticker_symbol}")

# Function to fetch news
def get_news(ticker_symbol):
    news_url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&sortBy=publishedAt&apiKey={news_api_key}"
    response = requests.get(news_url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data['articles']
    else:
        st.error("Error fetching news data.")
        return []

# Display news articles with sentiment analysis
articles = get_news(ticker_symbol)
if articles:
    for article in articles[:5]:  # Limit to 5 recent articles
        title = article['title']
        description = article['description']
        url = article['url']
        published_date = article['publishedAt']

        # Perform sentiment analysis on the article's title and description
        sentiment = TextBlob(description if description else title).sentiment
        sentiment_score = "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral"

        # Display news details
        st.markdown(f"**[{title}]({url})**")
        st.write(f"Published on: {published_date}")
        st.write(f"Sentiment: {sentiment_score}")
        st.write(description)
        st.write("---")
else:
    st.write("No recent news articles found.")
