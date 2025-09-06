import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.dates as mdates  
import plotly.graph_objects as go

# Initialize the app title
st.title('Stock Price Prediction App')

# List of commonly traded stock tickers for the dropdown
ticker_list = ['AMZN', 'MCD', 'NVDA', 'PINS', 'ETH-USD', 'BTC-USD','CSCO', 'INTC', 'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'GC=F', 'CL=F', 'SI=F', 'EREGL.IS', 'VESTL.IS', 'ODAS.IS', 'OSTIM.IS', 'VAKFN.IS', 'FENER.IS']

# User selects or inputs the ticker
ticker_input = st.text_input('Enter a ticker symbol:')
ticker_selection = st.selectbox('Or select from common tickers:', [''] + ticker_list)

# Use the manually entered ticker if available, otherwise use the dropdown
ticker = ticker_input.strip().upper() if ticker_input else ticker_selection

# Proceed if a ticker is entered or selected
if ticker:
    # Fetch historical data for the selected ticker
    data = yf.Ticker(ticker).history(period='max')

    # Ensure the data is not empty
    if not data.empty:
        st.write("Data Found!")
   
        data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
          # Add Parabolic SAR
        data['parabolic_sar'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close'], step=0.02, max_step=0.2).psar()

        # Add Stochastic Oscillator
        stochastic_oscillator = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['stoch_%K'] = stochastic_oscillator.stoch()
        data['stoch_%D'] = stochastic_oscillator.stoch_signal()

        # Selecting the necessary columns
        columns_to_keep = [
            'Close', 'Volume', 'trend_ema_fast', 'trend_ema_slow',
            'trend_macd', 'trend_macd_signal', 'trend_adx', 'momentum_rsi',
            'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volume_obv'
        ]

        # Prepare the selected data
        df_selected = data[columns_to_keep]
        df_selected = df_selected[-5*252:]  # Last 5 years of data
        df_selected = df_selected[df_selected['Volume'] != 0]

        # Display selected data
        st.subheader('Selected Data')
        st.dataframe(df_selected.head())

        # Prepare data for Prophet
        df_prophet = df_selected.reset_index()[['Date', 'Close']]
        df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        # Remove timezone from 'ds' column if present
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

        # Initialize and fit Prophet model
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)

        # Make future predictions
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        # Plot Prophet forecast
        st.subheader('Prophet Forecast for 1 Year')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        # Plot closing prices
        st.subheader('Close Price and Indicators')
        st.line_chart(df_selected['Close'])
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # EMA Fast and EMA Slow Plot
        plt.figure(figsize=(10, 5))
        plt.title('EMA Fast and EMA Slow')
        plt.plot(df_selected.index, df_selected['trend_ema_fast'], label='EMA Fast', color='blue')
        plt.plot(df_selected.index, df_selected['trend_ema_slow'], label='EMA Slow', color='red')
        plt.xlabel('Date')
        plt.ylabel('EMA Value')
        plt.legend()
        st.pyplot()

        # MACD and MACD Signal Plot
        plt.figure(figsize=(10, 5))
        plt.title('MACD and MACD Signal')
        plt.plot(df_selected.index, df_selected['trend_macd'], label='MACD', color='blue')
        plt.plot(df_selected.index, df_selected['trend_macd_signal'], label='MACD Signal', color='red')
        plt.xlabel('Date')
        plt.ylabel('MACD Value')
        plt.legend()
        st.pyplot()

            # Plot RSI
        st.subheader('RSI')
        st.line_chart(df_selected['momentum_rsi'])
        st.markdown('Overbought: RSI > 70')
        st.markdown('Oversold: RSI < 30')

        # Normalize the features for LSTM model
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_selected.drop(['Volume'], axis=1))

        # Define sequence length before using it
        sequence_length = 10  # Using last 10 days to predict the next day

        # Function to create sequences for LSTM input
        def create_sequences(data, sequence_length):
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data[i:(i + sequence_length), :])
                y.append(data[i + sequence_length, 0])  # Assuming 'Close' price is what we want to predict
            return np.array(X), np.array(y)

        # Generate sequences using the defined function
        X, y = create_sequences(scaled_data, sequence_length)
                
        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define the enhanced LSTM model
        model = Sequential([
            LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.3),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(100),
            Dropout(0.3),
            Dense(1)  
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=80,  # Increased the number of epochs to give the model more learning time
            batch_size=16,  # Adjust batch size as needed based on the model's performance and available compute resources
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Evaluate the model
        model.evaluate(X_test, y_test)

        # Make predictions
        predictions = model.predict(X_test)

        # Predict the next day's price using the LSTM model
        # Get the last sequence from the data
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = last_sequence.reshape((1, sequence_length, scaled_data.shape[1]))

        # Predict the next day
        next_day_prediction = model.predict(last_sequence)

        # Inverse transform the prediction to original scale
        dummy_next_day = np.ones((1, scaled_data.shape[1]))  # Create a dummy array with the same shape
        dummy_next_day[:, 0] = next_day_prediction[:, 0]  # Assuming the 'Close' price is the first column
        next_day_price = scaler.inverse_transform(dummy_next_day)[0, 0]

        # Example dummy data - assuming 'predictions' is your array of scaled predictions for the 'Close' price
        dummy_data = np.ones((len(predictions), scaled_data.shape[1]))  # Create a dummy array with same shape
        dummy_data[:, 0] = predictions[:, 0]  # Assuming the 'Close' price is the first column

        # Inverse transform the dummy data
        original_scale_data = scaler.inverse_transform(dummy_data)

        # Extract the 'Close' price column
        original_scale_close = original_scale_data[:, 0]

        # Assuming 'y_test' is in the scaled form and needs to be inversely transformed
        actual_prices = scaler.inverse_transform(np.ones((len(y_test), scaled_data.shape[1])) * y_test.reshape(-1, 1))

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                # Calculate RMSE and MAE
        rmse = np.sqrt(mean_squared_error(actual_prices[:, 0], original_scale_close))
        mae = mean_absolute_error(actual_prices[:, 0], original_scale_close)
        r_squared = r2_score(actual_prices[:, 0], original_scale_close)
        mape = mean_absolute_percentage_error(actual_prices[:, 0], original_scale_close)
        
        # Plotting the comparison with Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_selected.index[-len(actual_prices):], y=actual_prices[:, 0],
                                mode='lines', name='Actual Prices', line=dict(color='blue')))

        fig.add_trace(go.Scatter(x=df_selected.index[-len(original_scale_close):], y=original_scale_close,
                                mode='lines', name='Predicted Prices', line=dict(color='orange')))

        fig.update_layout(title='Comparison of Actual and Predicted Prices',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        xaxis=dict(rangeslider=dict(visible=True),
                                    type='date'))

        st.plotly_chart(fig)

        # Displaying the metrics
        st.subheader('Performance Metrics')
        st.write(f'R-squared: {r_squared:.2f}')
        st.write(f'Root Mean Squared Error: {rmse:.2f}')
        st.write(f'Mean Absolute Error: {mae:.2f}')
        st.write(f'Mean Absolute Percentage Error: {mape:.2f}%')  # Displaying MAPE

        st.subheader('Predicted Close Price for the Next Day')
        st.write(f'The predicted close price for {ticker} is  {next_day_price:.2f}')

    else:
        st.write("No data found for the selected ticker.")
else:
    st.write("Please select a ticker.")