import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime

model = joblib.load("trained_breakdown_classifier.pkl")

st.set_page_config(page_title="Medical Equipment IoT Monitoring", layout="wide")
st.title("🩺 Real-Time IoT Monitoring & Predictive Maintenance Dashboard")

device_mapping = {
    "ECG Monitor": 0,
    "Ventilator": 1,
    "Infusion Pump": 2,
    "Ultrasound Scanner": 3,
    "X-Ray Machine": 4,
    "Defibrillator": 5,
    "Patient Monitor": 6,
    "Anesthesia Machine": 7
}

device = st.selectbox("Select Medical Device", list(device_mapping.keys()))

sim_interval = st.sidebar.number_input("⏱ Update Interval (seconds)", min_value=1, max_value=10, value=2)
history_len = st.sidebar.number_input("📊 History Data Length", min_value=10, max_value=500, value=50)

if "history" not in st.session_state:
    st.session_state.history = []

placeholder = st.empty()

def generate_live_data():
    return {
        "usage_hours": np.random.uniform(100, 9000),
        "temperature": np.random.uniform(25, 90),
        "error_count": np.random.randint(0, 8)
    }

while True:
    data = generate_live_data()

    data["device_code"] = device_mapping[device]

    df = pd.DataFrame([data])[["device_code", "usage_hours", "temperature", "error_count"]]

  
    predicted_rul_days = model.predict(df)[0]
    predicted_rul_years = round(predicted_rul_days / 365, 2)


    if predicted_rul_years > 5:
        status = "✅ Healthy - No Action Needed"
        color = "green"
    elif 2 <= predicted_rul_years <= 5:
        status = "⚠️ Moderate - Plan Maintenance Soon"
        color = "orange"
    else:
        status = "🔴 Critical - Immediate Maintenance Required!"
        color = "red"

    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Usage Hours": round(data["usage_hours"], 2),
        "Temperature (°C)": round(data["temperature"], 2),
        "Errors": data["error_count"],
        "RUL (Years)": predicted_rul_years,
        "Status": status
    })

    # Trim history
    if len(st.session_state.history) > history_len:
        st.session_state.history.pop(0)

    # Display UI
    with placeholder.container():
        st.subheader(f"📟 Monitoring: **{device}**")
        st.write(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")

        col1, col2 = st.columns(2)
        col1.metric("Remaining Useful Life", f"{predicted_rul_years} Years")
        col2.markdown(f"<h3 style='color:{color};'>{status}</h3>", unsafe_allow_html=True)

        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

        st.line_chart(history_df[["Usage Hours", "Temperature (°C)", "RUL (Years)"]])

    time.sleep(sim_interval)
