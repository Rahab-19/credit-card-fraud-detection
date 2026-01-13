import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_transaction(transaction_df):
    scaled = scaler.transform(transaction_df)
    probability = model.predict_proba(scaled)[0][1]
    return probability

# Example usage
df = pd.read_csv("data/creditcard.csv")
sample = df.drop("Class", axis=1).iloc[[0]]

prob = predict_transaction(sample)
print("Fraud probability:", prob)
