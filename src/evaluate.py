import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# Load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Load model & scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

X_scaled = scaler.transform(X)

# Probabilities
probs = model.predict_proba(X_scaled)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y, probs)
print("ROC-AUC:", roc_auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y, probs)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, probs)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
