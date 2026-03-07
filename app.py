import streamlit as  st
st.title("Alzheimer's Disease Classification")
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("alzheimers_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])

def predict():

    data = request.json

    features = np.array(list(data.values())).reshape(1,-1)

    features = scaler.transform(features)

    prediction = model.predict(features)

    return jsonify({
        "Alzheimer Prediction": int(prediction[0])
    })


if __name__ == '__main__':
    app.run(debug=True)