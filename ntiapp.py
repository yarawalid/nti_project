import streamlit as st
import numpy as np
from tensorflow import keras

# UI settings
st.set_page_config(page_title="Stroke Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Stroke Risk Prediction App")

st.markdown(
    """
    Enter your health details below and check if you are at **risk of stroke**.  
    ⚕️ *This is a demo app — not medical advice.*
    """
)

# Load model
MODEL_PATH = "models/model.h5"

try:
    model = keras.models.load_model(MODEL_PATH)
    st.success(f"✅ Model loaded successfully! Input shape: {model.input_shape}")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Convert categorical inputs → numeric encoding (17 features)
def encode_inputs():
    gender_num = [1] if gender == "Male" else [0]
    hyper = [1] if hypertension == "Yes" else [0]
    heart = [1] if heart_disease == "Yes" else [0]
    married = [1] if ever_married == "Yes" else [0]

    work_types = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    work_onehot = [1 if work_type == wt else 0 for wt in work_types]

    residence_num = [1] if residence_type == "Urban" else [0]

    smoke_types = ["formerly smoked", "never smoked", "smokes", "Unknown"]
    smoke_onehot = [1 if smoking_status == sm else 0 for sm in smoke_types]

    features = [age, avg_glucose_level, bmi] \
        + gender_num + hyper + heart + married \
        + work_onehot + residence_num + smoke_onehot

    return np.array([features], dtype=np.float32), features

# Predict
if st.button("🔍 Predict"):
    try:
        input_data, feature_values = encode_inputs()
        st.write("📐 Encoded input shape:", input_data.shape)

        # Display input features in a table
        st.subheader("📋 Input Features")
        st.table({
            "Feature": [
                "Age", "Avg Glucose Level", "BMI", "Gender", "Hypertension",
                "Heart Disease", "Ever Married"
            ] + [f"WorkType_{w}" for w in ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]]
              + ["Residence_Urban"]
              + [f"Smoking_{s}" for s in ["formerly smoked", "never smoked", "smokes", "Unknown"]],
            "Value": feature_values
        })

        # Model prediction
        prediction = model.predict(input_data, verbose=0)
        score = float(prediction[0][0])
        st.write(f"🧮 Model raw output: **{score:.4f}**")

        if score > 0.5:
            st.error("⚠️ High risk of stroke")
        else:
            st.success("✅ Low risk of stroke")

    except Exception as e:
        st.error(f"Prediction error: {e}")

