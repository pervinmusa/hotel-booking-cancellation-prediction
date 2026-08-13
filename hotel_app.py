import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model and preprocessing
# =========================
model = joblib.load("xgboost_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Page
# =========================
st.title("🏨 Hotel Booking Cancellation Predictor")
st.write(
    "Enter the booking information below to predict whether "
    "the booking is likely to be canceled."
)

# =========================
# User Inputs
# =========================
no_of_adults = st.number_input(
    "Number of Adults",
    min_value=0,
    max_value=10,
    value=2
)
no_of_children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

# Safely get classes for selectboxes
meal_options = list(encoders["type_of_meal_plan"].classes_) if "type_of_meal_plan" in encoders else ["Meal Plan 1", "Meal Plan 2", "Not Selected"]
room_options = list(encoders["room_type_reserved"].classes_) if "room_type_reserved" in encoders else ["Room_Type 1", "Room_Type 2"]
market_options = list(encoders["market_segment_type"].classes_) if "market_segment_type" in encoders else ["Online", "Offline"]

type_of_meal_plan = st.selectbox(
    "Meal Plan",
    meal_options
)
room_type_reserved = st.selectbox(
    "Room Type",
    room_options
)
lead_time = st.number_input(
    "Lead Time (days)",
    min_value=0,
    max_value=500,
    value=30
)
market_segment_type = st.selectbox(
    "Market Segment",
    market_options
)
avg_price_per_room = st.number_input(
    "Average Price per Room",
    min_value=0.0,
    value=100.0
)
number_of_nights = st.number_input(
    "Number of Nights",
    min_value=1,
    max_value=30,
    value=2
)

# =========================
# Prediction
# =========================
if st.button("Predict"):
    # Automatically calculated
    total_guests = no_of_adults + no_of_children
    
    # Default values for less important inputs
    repeated_guest = 0
    no_of_previous_cancellations = 0
    no_of_previous_bookings_not_canceled = 0
    no_of_special_requests = 0
    season = "Summer"

    # Create input with ALL 14 features
    input_data = pd.DataFrame({
        "no_of_adults": [no_of_adults],
        "no_of_children": [no_of_children],
        "type_of_meal_plan": [type_of_meal_plan],
        "room_type_reserved": [room_type_reserved],
        "lead_time": [lead_time],
        "market_segment_type": [market_segment_type],
        "repeated_guest": [repeated_guest],
        "no_of_previous_cancellations": [no_of_previous_cancellations],
        "no_of_previous_bookings_not_canceled": [no_of_previous_bookings_not_canceled],
        "avg_price_per_room": [avg_price_per_room],
        "no_of_special_requests": [no_of_special_requests],
        "season": [season],
        "number_of_nights": [number_of_nights],
        "total_guests": [total_guests]
    })

    # Encode categorical columns safely
    for col in encoders:
        if col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col])

    # Make sure feature order is exactly the same
    feature_order = [
        "no_of_adults",
        "no_of_children",
        "type_of_meal_plan",
        "room_type_reserved",
        "lead_time",
        "market_segment_type",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "season",
        "number_of_nights",
        "total_guests"
    ]
    
    # Keep only available features in proper order
    existing_features = [f for f in feature_order if f in input_data.columns]
    input_data = input_data[existing_features]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    
    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Show result
    if prediction == 1:
        st.error("❌ The booking is likely to be CANCELED.")
    else:
        st.success("✅ The booking is likely to be NOT CANCELED.")
        
    st.write(f"Cancellation probability: **{probability:.2%}**")