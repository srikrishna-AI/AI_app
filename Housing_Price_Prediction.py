import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os
import numpy as np

# Title of the web app
st.title("House Price Prediction Web App")


# Step 1: Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('HousePricePrediction.csv')  # Replace with your file path
    return data


data = load_data()

# Step 2: Show the dataset to the user
if st.checkbox('Show Dataset'):
    st.write(data.head())


# Step 3: Handle missing values, categorical data, and preprocess
def preprocess_data(df):
    # Fill missing values with the median for numerical columns
    imputer = SimpleImputer(strategy='median')

    # Use LabelEncoder for categorical columns
    label_encoder = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')  # Fill NaN with 'Unknown' for categorical data
        df[col] = label_encoder.fit_transform(df[col])

    # Now apply the imputer on numerical columns
    df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))

    return df


# Preprocess the data
data = preprocess_data(data)

# Get the feature names (X) from the dataset, excluding the target column (SalePrice)
X = data.drop('SalePrice', axis=1)  # Adjust 'SalePrice' to your actual target column

# Step 4: Get user input for features, ensuring it matches the training data's feature set
st.subheader("Input the following details to predict house price:")

# Create user inputs dynamically based on the columns of X (training features)
user_input = {}
for column in X.columns:
    if X[column].dtype in [np.float64, np.int64]:
        user_input[column] = st.number_input(f'{column}', value=float(X[column].mean()))
    else:
        user_input[column] = st.text_input(f'{column}', value="Unknown")

# Convert the user input dictionary into a DataFrame and ensure it matches the training feature names
input_df = pd.DataFrame([user_input])


# Step 5: Train and save the model
def train_and_save_model():
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, data['SalePrice'], test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to a .pkl file using pickle
    with open('house_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    st.success("Model trained and saved as 'house_price_model.pkl'")


# Button to train and save the model
if st.button('Train and Save Model'):
    train_and_save_model()


# Step 6: Load the saved model
def load_model():
    if os.path.exists('house_price_model.pkl'):
        with open('house_price_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    else:
        st.error("Model file not found. Please train and save the model first.")


# Step 7: Make predictions with the loaded model
if st.button('Predict House Price'):
    model = load_model()
    if model:
        # Ensure the input features match the order and structure of the model's training features
        input_df = input_df[X.columns]
        prediction = model.predict(input_df)
        st.subheader(f'Predicted House Price: ${prediction[0]:,.2f}')
