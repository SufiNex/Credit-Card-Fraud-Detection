# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions using Python and machine learning algorithms.

## 📌 Project Overview

Credit card fraud is a major challenge in financial transactions. This project uses machine learning techniques to identify potentially fraudulent transactions and distinguish them from legitimate transactions.

## ✨ Features

- Credit card transaction data analysis
- Data cleaning and preprocessing
- Handling imbalanced data using SMOTE
- Feature scaling using StandardScaler
- Training and evaluating multiple ML models
- Fraud prediction with probability scores
- Interactive Streamlit web application
- Model performance visualization

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- Joblib

## 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

## 📊 Dataset

**European Credit Card Fraud Detection Dataset**

- Original records: 284,807
- Records after duplicate removal: 283,726
- Features: 30
- Target column: Class

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── App/
│   └── app.py
│
├── data/
│   └── creditcard.csv
│
├── Models/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── Notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb
│
├── Reports/
│
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Run Locally

Run the Streamlit application:

```bash
streamlit run App/app.py
```

## 🌐 Live Demo

**Try the deployed application here:**

[🚀 Credit Card Fraud Detection – Live Demo](https://credit-card-fraud-detection-sufi.streamlit.app/)
