import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer

# Load model
model = load("breast_cancer_model.joblib")

# Page settings
st.set_page_config(page_title="Breast Cancer Detector", layout="wide")
st.title("🩺 Breast Cancer Detector")

# Sidebar info/help
with st.sidebar:
    st.header("🧠 About")
    st.markdown("""
    This app predicts whether a breast tumor is **Malignant** or **Benign** using a trained ML model (SVM/Random Forest/KNN).
    
    **How to Use:**
    - Enter medical features in the input fields
    - Click **Predict**
    - See result and confidence visualized
    
    **Dataset**: [UCI Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)
    """)

# Load feature names
data = load_breast_cancer()
feature_names = data.feature_names

# Limit to top 10 features for simplicity
selected_features = feature_names[:10]
user_input = []

st.subheader("🔢 Input Features")

# Collect inputs
cols = st.columns(2)
for i, feature in enumerate(selected_features):
    with cols[i % 2]:
        value = st.number_input(f"{feature}", min_value=0.0, format="%.2f")
        user_input.append(value)

# Convert to DataFrame
input_df = pd.DataFrame([user_input], columns=selected_features)

# Predict
if st.button("🎯 Predict"):
    full_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    full_input[selected_features] = input_df[selected_features]

    prediction = model.predict(full_input)[0]
    prediction_proba = model.predict_proba(full_input)[0]

    result = "🔴 Malignant" if prediction == 0 else "🟢 Benign"
    st.markdown(f"### 🧬 Prediction: **{result}**")

    # Probability Gauge using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_proba[1]*100,
        delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        title={'text': "Benign Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if prediction == 1 else "red"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 100], 'color': "#ccffcc"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Raw probabilities
    st.markdown("#### 🔍 Prediction Confidence:")
    st.write(f"**Malignant**: `{prediction_proba[0]*100:.2f}%`")
    st.write(f"**Benign**: `{prediction_proba[1]*100:.2f}%`")
