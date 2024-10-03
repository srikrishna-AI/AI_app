import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Streamlit app setup
st.title("Stock Market Prediction App using LSTM")

# File upload
uploaded_file = st.file_uploader("Upload your stock market dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Select column with closing price
    column_name = st.selectbox("Select the column with closing prices:", df.columns)

    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = df[[column_name]].values
    scaled_data = scaler.fit_transform(data)

    # Split the data into training and testing
    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_size, :]
    test_data = scaled_data[training_size:, :]


    # Prepare input for the model
    def create_dataset(dataset, time_step=100):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)


    # Dynamically adjust the time_step based on the dataset size
    min_time_step = 10  # Minimum time steps allowed
    max_time_step = 100  # Default time step

    # Calculate the max possible time_step based on the dataset size
    max_possible_time_step = len(train_data) // 10  # Arbitrary logic: dataset length / 10

    # Use the smaller value between the max possible time step and the default
    time_step = min(max_time_step, max(min_time_step, max_possible_time_step))

    st.write(f"Using time_step: {time_step}")

    if len(train_data) > time_step:
        # Create training and testing datasets
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Ensure the data has enough samples to reshape
        if X_train.shape[0] > 0 and X_test.shape[0] > 0:
            # Reshape input data to be compatible with LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Define the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, batch_size=64, epochs=5)

            # Make predictions
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # Reverse scaling
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            # Prepare data for plotting
            full_data = np.concatenate((train_data, test_data), axis=0)
            full_data_inverse = scaler.inverse_transform(full_data)

            # Plot using Plotly (interactive with hover)
            st.subheader("Closing Price vs Time (Training set)")

            # Create a range for x-axis (time)
            time_range = np.arange(len(full_data_inverse))

            # Flatten the predictions to 1D
            train_predict_flat = train_predict.flatten()
            test_predict_flat = test_predict.flatten()

            # Padding with NaNs to align with the original data length
            train_predict_padded = np.concatenate(
                [train_predict_flat, [np.nan] * (len(full_data_inverse) - len(train_predict_flat))])
            test_predict_padded = np.concatenate([[np.nan] * training_size, test_predict_flat, [np.nan] * (
                        len(full_data_inverse) - len(test_predict_flat) - training_size)])

            # Plot using Plotly
            fig = go.Figure()

            # Add actual data
            fig.add_trace(go.Scatter(x=time_range, y=full_data_inverse.flatten(),
                                     mode='lines', name="Actual"))

            # Add train predictions
            fig.add_trace(go.Scatter(x=time_range, y=train_predict_padded,
                                     mode='lines', name="Train Predictions"))

            # Add test predictions
            fig.add_trace(go.Scatter(x=time_range, y=test_predict_padded,
                                     mode='lines', name="Test Predictions"))

            # Update layout
            fig.update_layout(title="Stock Price Predictions",
                              xaxis_title="Time",
                              yaxis_title="Price")

            # Display the plot in Streamlit
            st.plotly_chart(fig)

            # Predict future prices
            num_days = st.number_input("Enter number of future days to predict", min_value=1, max_value=100, value=30)

            # Last 100 days data to start prediction
            temp_input = test_data[-time_step:].tolist()

            if len(temp_input) < time_step:
                st.error("Not enough data to make predictions. Please try with a larger dataset.")
            else:
                # Flatten temp_input to avoid nested lists
                temp_input = [item for sublist in temp_input for item in sublist]

                # Predict future prices
                future_output = []
                for i in range(num_days):
                    if len(temp_input) > time_step:
                        x_input = np.array(temp_input[1:])
                        x_input = x_input.reshape(1, -1)
                        x_input = x_input.reshape((1, time_step, 1))
                        pred = model.predict(x_input, verbose=0)
                        temp_input.extend(pred[0].tolist())
                        temp_input = temp_input[1:]
                    else:
                        x_input = np.array(temp_input).reshape((1, time_step, 1))
                        pred = model.predict(x_input, verbose=0)
                        temp_input.extend(pred[0].tolist())

                    future_output.extend(pred.tolist())

                future_output = scaler.inverse_transform(future_output)

                # Convert predictions to DataFrame
                future_days = np.arange(1, num_days + 1)
                predictions_df = pd.DataFrame(future_output, columns=["Predicted Price"])
                predictions_df["Day"] = future_days

                # Plot future predictions using Plotly
                st.subheader(f"Prediction for the next {num_days} days")

                fig2 = go.Figure()

                # Add future predictions
                fig2.add_trace(go.Scatter(x=future_days, y=future_output.flatten(),
                                          mode='lines', name="Future Predictions"))

                # Update layout
                fig2.update_layout(title=f"Stock Price Prediction for Next {num_days} Days",
                                   xaxis_title="Days",
                                   yaxis_title="Predicted Price")

                # Display the plot in Streamlit
                st.plotly_chart(fig2)

                # Option to download predictions as CSV
                csv = predictions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name='stock_predictions.csv',
                    mime='text/csv',
                )
        else:
            st.error("Not enough data to reshape. Please try with a larger dataset.")
else:
    st.write("Please upload a dataset to begin.")
