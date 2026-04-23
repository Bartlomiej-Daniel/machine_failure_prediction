import streamlit as st
import sys
from pathlib import Path
import joblib
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

model_path = base_dir / "models"

# Layout
st.set_page_config(page_title="Machine Failure Prediction", layout="wide")

st.title("Machine Failure Prediction")
st.write("Enter the parameters and let the model predict whether the machine will fail.")

st.sidebar.title("Model Info")

st.sidebar.markdown("### Random Forest")
st.sidebar.metric("Precision:", 0.9)
st.sidebar.metric("Recall:", 0.63)

air_temp = st.number_input("Air temperature [K]", value=300.0)
process_temp = st.number_input("Process temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool wear [min]", value=100.0)


input_data = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])


model = joblib.load(model_path / "rf_model.pkl")

prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0][1]

if st.button("Predict"):
    st.write(f"Failure probability: {proba:.2f}")
    
    if prediction == 1:
        st.error("Machine will FAIL")
    else:
        st.success("Machine is OK")
