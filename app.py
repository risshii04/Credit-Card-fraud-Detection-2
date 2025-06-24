import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

# App title
st.title("💳 Credit Card Fraud Detection")

# Instructions
st.write("Enter transaction details to check for potential fraud.")

# Create input fields for V1 to V28 and Amount
input_data = {}
for i in range(1, 29):
    input_data[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f")

# Amount input
input_data['Amount'] = st.number_input("Amount", value=0.0, format="%.2f")

# Predict button
if st.button("Predict"):
    # Convert input to NumPy array and reshape
    features = np.array([list(input_data.values())]).astype(np.float32)

    # Predict probability and class
    prob = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    # Output
    st.write(f"**Prediction:** {'⚠️ Fraud' if prediction == 1 else '✅ Not Fraud'}")
    st.write(f"**Probability of Fraud:** {prob:.4f}")
