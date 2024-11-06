import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('cardekho.csv')  # Ensure the path is correct
    return data

# Preprocessing function
def preprocess_data(data, input_features):
    # Filter data to include only the selected input features
    X = data[input_features]
    y = data['selling_price']

    # Encoding categorical features
    categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[categorical_columns])

    # Add encoded features back to the DataFrame, dropping original categorical columns
    X = np.concatenate([X.drop(categorical_columns, axis=1), X_encoded], axis=1)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler

# Main function for the Streamlit app
def main():
    st.title('Car Price Prediction App')

    # Load data
    data = load_data()

    # Display the first few rows of the dataset
    st.write("### Dataset Preview")
    st.dataframe(data.head())  # Display the first 5 rows of the dataset

    # Define input features
    input_features = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']

    # Preprocess the data
    X_train, X_test, y_train, y_test, encoder, scaler = preprocess_data(data, input_features)

    # Train a simple model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse}')

    # User input for new prediction
    st.sidebar.header('Enter car details for prediction')

    year = st.sidebar.number_input('Year of the car', min_value=1990, max_value=2024, value=2015)
    km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, max_value=500000, value=50000)
    fuel = st.sidebar.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    seller_type = st.sidebar.selectbox('Seller Type', options=['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox('Transmission Type', options=['Manual', 'Automatic'])
    owner = st.sidebar.selectbox('Owner Type', options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

    # Encode user input
    input_data_encoded = encoder.transform([[fuel, seller_type, transmission, owner]])

    # Prepare the data for prediction
    input_data = np.array([[year, km_driven]])
    input_data = np.concatenate([input_data, input_data_encoded], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    predicted_price = model.predict(input_data_scaled)
    st.write(f'Predicted Selling Price: {predicted_price[0]:.2f}')

if __name__ == '__main__':
    main()
