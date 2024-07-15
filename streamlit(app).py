import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define user credentials (this is a simple example; use a more secure method for production)
USER_CREDENTIALS = {"senthalan": "password"}

# Function to check login
def check_login(username, password):
    return USER_CREDENTIALS.get(username) == password

# Streamlit login page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
            st.experimental_rerun()  # Refresh to show the main app
        else:
            st.error("Invalid username or password")

# Streamlit help content
def show_help():
    st.subheader("Help Section")
    st.write("""
    **Intelligent Stock Trader (IST) Platform Help**

    - **Login**: Enter your username and password to log in.
    - **Stock Symbol**: Input the ticker symbol of the stock you want to analyze.
    - **Date Range**: The app will load historical data from 2019-01-01 to 2023-01-01.
    - **ARIMA Model**: The app uses ARIMA to forecast stock prices and selects the best model based on AIC.
    - **Linear Regression**: Linear regression is used to predict future prices based on historical data.
    - **Forecasting**: View predictions from both ARIMA and Linear Regression models.
    - **Transaction Suggestions**: Input desired profit margin and select prediction interval to get transaction suggestions.
    - **Plotting**: Visualize historical data, forecasts, and suggested transactions.

    For further assistance, please contact support or refer to the documentation.
    """)

# Streamlit main app page
def main_page():
    # Streamlit App Title
    st.title('Intelligent Stock Trader (IST) Platform')

    # Input for Stock Symbol
    st.write("Enter the stock symbol:")
    stock_symbol = st.text_input("Stock Symbol", "AAPL")

    
    # Date Range for historical data
    START = "2019-01-01"
    END = "2023-01-01"

    # Load Data
    @st.cache_data
    def load_data(ticker):
        stock_data = yf.Ticker(ticker)
        historical_data = stock_data.history(period="1d", start=START, end=END)
        historical_data = historical_data[['Close']].dropna()
        historical_data.index = pd.to_datetime(historical_data.index)
        historical_data = historical_data.asfreq('B').fillna(method='ffill')
        return historical_data

    data_load_state = st.text('Loading data...')
    data = load_data(stock_symbol)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data(data):
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Stock Price Over Time')
        st.pyplot(fig)

    plot_raw_data(data)

    # ARIMA Model Training and Forecast
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    @st.cache_resource
    def get_best_arima_model(data, p_values, d_values, q_values):
        best_aic = np.inf
        best_order = None
        best_model = None

        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(data, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue
        return best_order, best_model 

    p_values = range(0, 6)
    d_values = range(0, 3)
    q_values = range(0, 6)

    best_order, best_model = get_best_arima_model(train, p_values, d_values, q_values)
    st.write(f'Best ARIMA order: {best_order}')
    st.write(best_model.summary())

    # Forecasting ARIMA
    arima_prediction = best_model.forecast(steps=len(test))
    test['arima_predictions'] = arima_prediction.values

    # Prepare data for Linear Regression
    def prepare_linear_regression_data(data):
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
        X = data[['Date']]
        y = data['Close']
        return X, y

    def train_linear_regression(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def predict_with_linear_regression(model, X):
        return model.predict(X)

    # Prepare data for training and testing Linear Regression
    X_train, y_train = prepare_linear_regression_data(train)
    X_test, y_test = prepare_linear_regression_data(test)

    linear_model = train_linear_regression(X_train, y_train)

    # Make Predictions with Linear Regression
    test['lr_predictions'] = predict_with_linear_regression(linear_model, X_test)

    # Plotting Forecasts
    def plot_forecast(train, test):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train.index, train['Close'], label='Training Data')
        ax.plot(test.index, test['Close'], label='Actual prices')
        ax.plot(test.index, test['arima_predictions'], label='ARIMA Predictions', color='red')
        ax.plot(test.index, test['lr_predictions'], label='Linear Regression Predictions', color='green')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend()
        st.pyplot(fig)

    st.subheader('Forecast data')
    st.write(test.tail())

    st.subheader('Forecast Plot')
    plot_forecast(train, test)

    # Transaction Suggestion
    profit_margin = st.number_input('Enter desired profit margin (£)', min_value=0, value=50)
    interval = st.selectbox('Select prediction interval', ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

    @st.cache_resource
    def suggest_transactions(data, profit_margin, interval):
        suggestions = []
        if interval == 'Daily':
            periods = 1
        elif interval == 'Weekly':
            periods = 5
        elif interval == 'Monthly':
            periods = 20
        else:
            periods = 60

        for i in range(len(data) - periods):
            buy_price = data['Close'].iloc[i]
            sell_price = data['Close'].iloc[i + periods]
            profit = (sell_price - buy_price) * 1  # Assuming 1 share
            if profit >= profit_margin:
                suggestions.append((data.index[i], data.index[i + periods], profit))

        return suggestions

    transactions = suggest_transactions(data, profit_margin, interval)

    st.subheader('Suggested Transactions')
    for trans in transactions:
        st.write(f'Buy on {trans[0]} and sell on {trans[1]} for a profit of £{trans[2]:.2f}')

    # Help Button
    if st.button('Help'):
        show_help()

# Streamlit app logic
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    login_page()
else:
    main_page()
