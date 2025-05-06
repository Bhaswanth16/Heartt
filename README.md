
# 🫀 Cardiovascular Disease Risk Prediction App

This Streamlit web app uses a trained **Gradient Boosting Machine (GBM)** model to predict the risk of **cardiovascular disease (CVD)** based on a user's health and lifestyle inputs.

## 🚀 How It Works

Users enter health indicators (like BMI, exercise, smoking, diabetes, etc.) into a form. The model returns a prediction of whether they are likely at **high risk** or **low risk** for cardiovascular disease.

### ✅ Features Used
- Age Group
- BMI
- Days of Poor Physical & Mental Health
- Exercise Activity
- Smoking History
- Alcohol Consumption
- Diabetes Status
- Stroke History
- Mammogram History
- COVID Status & Symptoms

## 🧠 Model Details
- Trained on a filtered version of the BRFSS dataset
- Balanced using oversampling for better recall
- Implemented using `GradientBoostingClassifier` from scikit-learn

## 📁 Repository Contents

```
📦 cvd-prediction-app/
├── app.py               # Streamlit UI
├── cvd_model.pkl        # Trained Gradient Boosting model
├── feature_list.pkl     # Feature order for model input
└── requirements.txt     # Required Python packages
```

## 📦 Requirements

Create a `requirements.txt` with:

```
streamlit
scikit-learn
numpy
```




## ⚠️ Disclaimer

This application is for educational purposes only and **does not replace professional medical advice**.
