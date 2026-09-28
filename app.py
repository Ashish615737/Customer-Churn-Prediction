
import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.pkl")


model = load_model()


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter the customer's details below to predict whether "
    "the customer is likely to churn."
)


# ---------------------------------------------------
# Customer Information
# ---------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )


with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# ---------------------------------------------------
# Payment and Charges
# ---------------------------------------------------

st.header("Payment & Charges")

col4, col5, col6 = st.columns(3)


with col4:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col5:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


with col6:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=monthly_charges * tenure,
        step=10.0
    )


# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    # Create input dataframe
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Get probability
    probability = model.predict_proba(input_data)[0][1]


    # ---------------------------------------------------
    # Display Result
    # ---------------------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Customer is likely to CHURN")

        st.write(
            f"Estimated churn probability: "
            f"**{probability * 100:.2f}%**"
        )

    else:

        st.success("✅ Customer is likely to STAY")

        st.write(
            f"Estimated churn probability: "
            f"**{probability * 100:.2f}%**"
        )


    # ---------------------------------------------------
    # Display Input Data
    # ---------------------------------------------------

    with st.expander("View Customer Information"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
