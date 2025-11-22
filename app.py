import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the model
model = pickle.load(open("linearmodel.pkl", "rb"))

st.title("Price Prediction App")

st.write("Enter the features to predict the price:")

df = pd.read_csv("clean_data.csv")   # put your correct dataset name
locations = sorted(df['location'].unique())

# Create input boxes
location= st.selectbox("Location", locations)
total_sqft = st.number_input("total_sqft", min_value=1, max_value=100000, step=1)
bath = st.number_input("bath", min_value=1, max_value=20, step=1)
bhk=st.number_input("bhk", min_value=1, max_value=20, step=1)


if st.button("Predict Price"):
    # Convert input to array
    input_df = pd.DataFrame({
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "bhk": [bhk]
    })
    pred = model.predict(input_df)

    st.success(f"Predicted Price: {pred[0]:.2f}")
