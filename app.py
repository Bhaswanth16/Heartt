
import streamlit as st
import numpy as np
import pickle

# Load model and features
with open("cvd_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("feature_list.pkl", "rb") as f:
    feature_list = pickle.load(f)

st.set_page_config(page_title="CVD Risk Predictor", page_icon="🫀")
st.title("🫀 Cardiovascular Disease (CVD) Risk Predictor")
st.markdown("This tool uses a trained machine learning model to estimate your risk of cardiovascular disease based on your health indicators.")

st.header("📝 Enter your health information below:")

# Input form
with st.form("cvd_form"):
    age = st.slider("Age Group (1 = 18–24 ... 13 = 80+)", 1, 13, 5)
    bmi = st.number_input("BMI (e.g., 25.0)", 10.0, 50.0, step=0.1)
    bmi = int(bmi * 100)  # convert to _BMI5 format
    
    physhlth = st.slider("Number of days in poor physical health (last 30 days)", 0, 30, 5)
    menthlth = st.slider("Number of days in poor mental health (last 30 days)", 0, 30, 5)
    exerany2 = st.radio("Do you engage in physical activity or exercise?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    smoke100 = st.radio("Have you smoked at least 100 cigarettes in your life?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    smokday2 = st.radio("Do you currently smoke?", [1, 2, 3], format_func=lambda x: {1: "Every day", 2: "Some days", 3: "Not at all"}[x])
    alcday4 = st.slider("Number of days you consumed alcohol in the past 30 days", 0, 30)
    diabete4 = st.radio("Do you have diabetes?", [1, 2, 3], format_func=lambda x: {1: "Yes", 2: "No", 3: "Pre-diabetic"}[x])
    cvdstrk3 = st.radio("Have you ever had a stroke?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    hadmam = st.radio("Have you ever had a mammogram?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    covidpos = st.radio("Have you tested positive for COVID-19?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    covidsmp = st.radio("Did you experience COVID symptoms?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    covidprm = st.radio("Do you still experience COVID symptoms?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    input_data = np.array([[age, bmi, physhlth, menthlth, exerany2, smoke100, smokday2,
                            alcday4, diabete4, cvdstrk3, hadmam, covidpos, covidsmp, covidprm]], dtype=np.float32)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error("⚠️ You may be at high risk of cardiovascular disease.\nConfidence: {:.2f}".format(probability))
    else:
        st.success("✅ You appear to be at low risk of cardiovascular disease.\nConfidence: {:.2f}".format(1 - probability))

    st.markdown("---")
    st.markdown("**Disclaimer:** This prediction is based on a statistical model and should not replace professional medical advice.")
