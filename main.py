import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. Page Configuration & Styling
st.set_page_config(page_title="Financial Risk & Anomaly AI", page_icon="💹", layout="wide")

st.title("💹 AI-Powered Financial Risk & Fraud Anomaly Detection")
st.markdown("### Real-Time Transaction Monitoring & Market Volatility Engine")

# 2. Sidebar for Financial Parameters
st.sidebar.header("💳 Transaction & Market Parameters")
txn_amount = st.sidebar.number_input("Transaction Amount ($)", min_value=1.0, max_value=100000.0, value=250.0)
transaction_velocity = st.sidebar.slider("Transactions in Last 1 Hour", 1, 50, 2)
account_age_days = st.sidebar.slider("Account Age (Days)", 1, 3650, 365)
geo_distance_km = st.sidebar.slider("Distance from Home Location (km)", 0, 5000, 10)

# 3. Training an Unsupervised ML Model (Isolation Forest for Anomaly/Fraud Detection)
np.random.seed(42)
n = 1000
X_mock = np.column_stack([
    np.random.exponential(scale=300.0, size=n),
    np.random.poisson(lam=3, size=n),
    np.random.randint(30, 3650, size=n),
    np.random.exponential(scale=20.0, size=n)
])

# Using Isolation Forest (Unsupervised ML for finding outliers/frauds)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_mock)

# 4. Main Layout (Dashboard Grid)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live Transaction Grid Log")

    input_df = pd.DataFrame({
        'Transaction Amount ($)': [txn_amount],
        'Txn Velocity (1hr)': [transaction_velocity],
        'Account Age (Days)': [account_age_days],
        'Geo Distance (km)': [geo_distance_km]
    })

    # Display Data in Grid Format
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("🛡️ AI Security & Risk Verdict")
    if st.button("Evaluate Transaction Risk"):
        # Isolation Forest predicts: 1 for normal, -1 for anomaly/fraud
        prediction = model.predict(input_df)
        score = model.decision_function(input_df)[0]  # Lower score means more anomalous

        if prediction[0] == -1:
            st.error(
                f"🚨 HIGH RISK: Potential Financial Fraud Detected!\n\nAnomaly Score: {score:.4f}\n\n*Action: Flagged for manual security review.*")
        else:
            st.success(f"✅ Secure: Legitimate Transaction.\n\nAnomaly Score: {score:.4f}")

# Footer System Information
st.markdown("---")
st.text("Fintech ML Core v3.5 | Algorithm: Isolation Forest (Unsupervised) | Status: Active Monitoring")