# Credit Card Fraud Detection using Machine Learning

##  Project Overview
This project builds an end-to-end machine learning system to detect fraudulent credit card transactions.  
The dataset is highly imbalanced, with fraudulent transactions representing a very small fraction of all transactions, which makes fraud detection a challenging real-world problem.

The goal is to accurately identify fraudulent transactions while minimizing missed fraud cases (false negatives).


##  Problem Statement
Credit card fraud causes significant financial losses for banks and customers.  
Traditional rule-based systems struggle to detect new and evolving fraud patterns.

This project applies **supervised machine learning** techniques to:
- Learn fraud patterns from historical transaction data
- Handle extreme class imbalance
- Evaluate models using fraud-appropriate metrics

---

##  Dataset
- **Source:** Kaggle Credit Card Fraud Dataset  
- **Transactions:** 284,807  
- **Fraud cases:** 492 (≈0.17%)  
- **Features:**  
  - `V1–V28`: PCA-transformed features  
  - `Time`: Time elapsed since first transaction  
  - `Amount`: Transaction amount  
  - `Class`: Target variable (0 = Normal, 1 = Fraud)

---

##  Technologies Used
- Python  
- pandas, numpy  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- matplotlib, seaborn  
- joblib  
- VS Code  

---

##  Project Structure
credit Card ML/
│
├── data/
│ └── creditcard.csv
│
├── notebooks/
│ └── fraud_detection.ipynb
│
├── src/
│ ├── train_logistic.py
│ ├── train_random_forest.py
│ ├── evaluate.py
│ └── predict.py
│
├── models/
│ ├── fraud_model.pkl
│ └── scaler.pkl
│
├── requirements.txt
└── README.md

## 🔍 Machine Learning Approach

### 1 Data Preprocessing
- Removed target variable (`Class`) from features
- Applied **StandardScaler** to normalize feature values
- Used **Stratified Train-Test Split** to preserve fraud ratio

### 2 Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**  
- Balanced the training dataset to improve fraud detection recall

### 3 Models Trained
- **Logistic Regression** (baseline, interpretable)
- **Random Forest Classifier** (captures non-linear fraud patterns)

### 4 Model Evaluation
Evaluated models using:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Score
- Precision–Recall Curve

> Recall was prioritized to minimize missed fraud cases.

### 5 Threshold Optimization
- Adjusted prediction thresholds to improve fraud detection performance
- Demonstrates business-aware ML decision-making

##  Results
- Random Forest outperformed Logistic Regression
- Improved fraud recall while maintaining reasonable precision
- ROC-AUC used to assess overall discriminative power

##  Single Transaction Prediction
The project supports predicting fraud probability for an individual transaction, simulating real-time fraud detection.

##  How to Run the Project

### 1. Install dependencies

   pip install -r requirements.txt

## 2. Train models
   python src/train_logistic.py
   python src/train_random_forest.py

### 3. Evaluate model
    python src/evaluate.py

### 4. Predict a transaction
    python src/predict.py

### Future Improvements

Hyperparameter tuning

XGBoost / LightGBM models

Cost-sensitive learning

Model deployment with Streamlit or FastAPI

Real-time fraud monitoring

### Author

Rahab Modiba
Aspiring Data Scientist | Pharmacist Assistant transitioning into Data & Machine Learning