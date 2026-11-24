# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Project Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Model Loading
try:
    model = joblib.load(
        BASE_DIR / "Models" / "fraud_detection_model.pkl"
    )

    scaler = joblib.load(
        BASE_DIR / "Models" / "scaler.pkl"
    )

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()


# App Header
st.title("💳 Credit Card Fraud Detection")

st.write(
    "A Machine Learning application to identify "
    "fraudulent and legitimate credit card transactions."
)

st.success("Model Loaded Successfully!")

st.divider()


# Performance Dashboard
st.header("📊 Model Performance Dashboard")

report_path = BASE_DIR / "Reports" / "model_performance.csv"

if report_path.exists():

    try:
        performance_df = pd.read_csv(report_path)

        st.subheader("Model Comparison")

        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True
        )

        # F1 Score Chart
        if {"Model", "F1 Score"}.issubset(performance_df.columns):

            st.subheader("F1 Score Comparison")

            f1_chart = performance_df.set_index("Model")[
                "F1 Score"
            ]

            st.bar_chart(f1_chart)

        # ROC-AUC Chart
        if {"Model", "ROC-AUC"}.issubset(performance_df.columns):

            st.subheader("ROC-AUC Comparison")

            roc_chart = performance_df.set_index("Model")[
                "ROC-AUC"
            ]

            st.bar_chart(roc_chart)

    except Exception as e:
        st.warning(f"Dashboard Error: {e}")

else:
    st.warning(
        "Performance report not found. "
        "Please save model results from your notebook."
    )


st.divider()


# Transaction Prediction
st.header("🔍 Fraud Detection")

st.write("Enter transaction details below.")

with st.form("fraud_detection_form"):

    st.subheader("Basic Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=0.0
        )

    with col2:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

    st.subheader("Transaction Features")

    features = {}

    cols = st.columns(2)

    for i in range(1, 29):

        with cols[(i - 1) % 2]:

            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f"
            )

    submit = st.form_submit_button(
        "🔍 Detect Fraud",
        use_container_width=True
    )


# Prediction
if submit:

    try:
        # Prepare Input
        input_data = {
            "Time": time,
            **features,
            "Amount": amount
        }

        feature_columns = (
            ["Time"]
            + [f"V{i}" for i in range(1, 29)]
            + ["Amount"]
        )

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Apply Scaling
        input_df[["Time", "Amount"]] = scaler.transform(
            input_df[["Time", "Amount"]]
        )

        # Model Prediction
        prediction = model.predict(input_df)[0]

        fraud_score = model.predict_proba(input_df)[0][1]

        # Display Result
        st.divider()

        st.header("📋 Prediction Result")

        st.metric(
            label="Model Fraud Score",
            value=f"{fraud_score * 100:.2f}%"
        )

        st.progress(float(fraud_score))

        if prediction == 1:

            st.error(
                "⚠️ Fraudulent Transaction Detected!"
            )

        else:

            st.success(
                "✅ Legitimate Transaction"
            )

        st.caption(
            "The fraud score is the model's estimated "
            "class probability, not a guaranteed real-world risk."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# Footer
st.divider()

st.caption(
    "Credit Card Fraud Detection | Machine Learning Project"
)
