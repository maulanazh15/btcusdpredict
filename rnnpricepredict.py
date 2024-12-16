# Import pandas
import pandas as pd
import datetime
# Import yfinance
import yfinance as yf
import uuid
# import time
import time as t
import pytz
# Import the required libraries
from keras._tf_keras.keras.models import  load_model
from sklearn.preprocessing import MinMaxScaler 
import numpy as np

# Imports app
import plotly.graph_objects as go
import streamlit as st

# Import evaluate library
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, mean_absolute_error

# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]

# Period must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

class BitcoinData :
    # Function to return lookback
    def setLookback(interval) :
        if interval=="5m":
            return int(24*60/5/2)
        elif interval=="1h":
            return int(24/2)
        elif interval=="1d":
            return int(7)
    
    # Function to fetch the crypto history
    def fetchRealTimeData(crypto_ticker, period, interval, lookback):
        # Try to generate the predictions
        try:
            # Pull the data for the first security
            crypto_data = yf.Ticker(crypto_ticker)

            # Extract the data for last 1yr with 1d interval
            crypto_data_hist = crypto_data.history(period=period, interval=interval)

            # Clean the data for to keep only the required columns
            crypto_data_close = crypto_data_hist[["Close"]]

            # Fill missing values
            crypto_data_close = crypto_data_close.ffill()

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(crypto_data_close)

            x_data = []
            y_data = []
            
            # lookback = 5
            # [30, 31, 94, 50, 27, 68, 37, 69, 27, 57, 25, 48, 50]
            # train data  =   [30, 31, 94, 50, 27] => 68
            #                 [31, 94, 50, 27, 68] => 37
            #                 [94, 50, 27, 68, 37] => 69
            #                 ...
            #                 [69, 27, 57, 25, 48] => 50
            # x train data = ([30, 31, 94, 50, 27], [31, 94, 50, 27, 68], [94, 50, 27, 68, 37], ..., [69, 27, 57, 25, 48])
            # y train data = (68, 37, 69, 27, 57, 25, 48, 50)

            for i in range(lookback,len(scaled_data)):
                x_data.append(scaled_data[i-lookback:i,0])
                y_data.append(scaled_data[i,0])

            # Converting the x and y values to numpy arrays
            x_data, y_data = np.array(x_data), np.array(y_data)

            # Reshaping x and y data to make the calculations easier
            x_data = np.reshape(x_data, (x_data.shape[0],x_data.shape[1],1))
            y_data = np.reshape(y_data, (y_data.shape[0],1))
            
            history_df = crypto_data_close

            # Return the required data
            return history_df

        # If error occurs
        except:
            # Return None
            return None

class Prediction :
    # Define lookback and load the model from the file
    def setModel(interval) :
        if interval=="5m":
            return load_model("rnnbtc5min.keras")
        elif interval=="1h":
            return load_model("rnnbtc1h.keras")
        elif interval=="1d":
            return load_model("rnnbtc1d.keras")

    # Function to generate the crypto prediction
    def runPredictionModel(model, history_df, time, interval, lookback):
        # Try to generate the predictions
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(history_df)
            closing_prices = scaled_data[len(scaled_data)-lookback:]
            closing_prices = closing_prices.reshape(closing_prices.shape[0], 1)
            current_batch = closing_prices.reshape(1, lookback, 1)
            current_batch = current_batch[:,:,:]

            future_predictions = []
            for i in range(time):  # Predicting range time
                # Get the prediction (next day)
                next_prediction = model.predict(current_batch)
                # print(scaler.inverse_transform(next_prediction))
                # Reshape the prediction to fit the batch dimension
                next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
                
                # Append the prediction to the batch used for predicting
                current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
                
                # Inverse transform the prediction to the original price scale
                future_predictions.append(scaler.inverse_transform(next_prediction)[0, 0])

            last_time = history_df.index[-1]
            freq = interval
            if interval=="5m":
                next_time = last_time + pd.Timedelta(minutes=5)
                freq = "5min"
            elif interval=="1h":
                next_time = last_time + pd.Timedelta(hours=1)
            elif interval=="1d":
                next_time = last_time + pd.Timedelta(days=1)
            future_prediction_times = pd.date_range(start=next_time, periods=len(future_predictions), freq=freq)
            future_predictions_df = pd.DataFrame(future_predictions, index=future_prediction_times, columns=["Close"])

            # Return the required data
            return future_predictions_df

        # If error occurs
        except:
            # Return None
            return None

class Dashboard :
    def run():
        # Configure the page
        st.set_page_config(
            page_title="Bitcoin Price Prediction",
            page_icon="₿📈",
        )

        #####Sidebar Start#####

        # Add a sidebar
        st.sidebar.markdown("## **User Input Features**")
        
        # Create dictionary for periods and intervals
        intervals = {
            "5m": "5d",
            "1h": "6mo",
            "1d": "10y",
        }

        # Add a selector for interval
        st.sidebar.markdown("### **Select interval**")
        interval = st.sidebar.selectbox("Choose an interval", list(intervals.keys()))

        # Build the crypto ticker
        crypto_ticker = "BTC-USD"
        period = intervals[interval]
        lookback = BitcoinData.setLookback(interval)
        model = Prediction.setModel(interval)
        
        # Add an period of future predictions
        st.sidebar.markdown("### **Predictions period**")
        time = st.sidebar.number_input(label="Time of interval",min_value=1, max_value=lookback)

        #####Sidebar End#####
        placeholder = st.empty()
        
        for seconds in range(24*60) :
            crypto_data = yf.Ticker(crypto_ticker)
            price = crypto_data.history(interval="1m", period="1d")['Close'].map('{:,.2f}'.format)
            price.index = price.index.tz_convert('Asia/Jakarta')
            date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Unpack the data
            history_df = BitcoinData.fetchRealTimeData(crypto_ticker, period, interval, lookback).map('{:.2f}'.format)
            future_predictions_df = Prediction.runPredictionModel(model, history_df, time, interval, lookback)
            # Check if the data is not None
            if history_df is not None and future_predictions_df is not None:
                # Add a title to the crypto prediction graph
                future_predictions_df.index = future_predictions_df.index.tz_convert('Asia/Jakarta')
                future_predictions_df['Formatted_Datetime'] = future_predictions_df.index.strftime('%A, %Y-%m-%d %H:%M:%S')
                history_df.index = history_df.index.tz_convert('Asia/Jakarta')
                # history_df['Close'] = history_df['Close'].map('{:,.2f}'.format)
                future_predictions_df['Close'] = future_predictions_df['Close'].map('{:,.2f}'.format)
                with placeholder.container() : 
                    # Create a plot for the crypto prediction
                    st.markdown(f"## **Bitcoin Price : $ {price[-1]} (few minute delay)** \n Price Time : {(price.index[-1].strftime('%A, %Y-%m-%d %H:%M:%S'))}")
                    st.markdown(f"**Time Now: ({date_now})**")
                    st.markdown("**Bitcoin Prediction**")
        
                    # st.plotly_chart(fig, use_container_width=True)

                    # Create data table of future
                    fig = go.Figure(
                        data=[
                            go.Table(
                                header=dict(values=["Datetime", "Close"], align='left'),
                                cells=dict(values=[future_predictions_df['Formatted_Datetime'], future_predictions_df['Close']], align='left')
                            ),
                        ]
                    )
                    # Generate a random key for this table
                    random_key_chart_2 = str(uuid.uuid4())
                    st.plotly_chart(fig, use_container_width=True, key=random_key_chart_2)
                    # st.plotly_chart(fig, use_container_width=True)
                    fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=history_df.index,
                                y=history_df["Close"],
                                name="History",
                                mode="lines",
                                line=dict(color="blue"),
                            ),
                            go.Scatter(
                                x=future_predictions_df.index,
                                y=future_predictions_df["Close"],
                                name="Future Predictions",
                                mode="lines",
                                line=dict(color="red"),
                            ),
                        ]
                    )

                    # Customize the crypto prediction graph
                    fig.update_layout(xaxis_rangeslider_visible=True)
                    # Generate a random key for this chart
                    random_key_chart_1 = str(uuid.uuid4())
                    st.plotly_chart(fig, use_container_width=True, key=random_key_chart_1)


            # If the data is None
            else:
                # Add a title to the crypto prediction graph
                st.markdown("## **Bitcoin Prediction**")

                # Add a message to the crypto prediction graph
                st.markdown("### **No data available**")

            #####Bitcoin Prediction Graph End#####
            t.sleep(1)

if __name__ == '__main__': 
    # dashboard = Dashboard()
    # dashboard.run()
    Dashboard.run()
        
        
