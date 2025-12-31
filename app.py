# app.py - Ultra Simple Demo Version
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cyber Security Demo", layout="wide")
st.title("🛡️ Cybersecurity Attack Classifier (Demo)")
st.markdown("---")

# Create demo data
np.random.seed(42)
feature_names = [f'Feature_{i}' for i in range(1, 11)]
attack_types = ['Normal', 'DDoS', 'Port Scan', 'SQL Injection', 'Malware']

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["Demo Model 1", "Demo Model 2"])
    st.slider("Confidence Threshold", 0.0, 1.0, 0.7)

st.header("📝 Enter Network Traffic Features")

# Create input columns
cols = st.columns(5)
inputs = {}

for i, feature in enumerate(feature_names):
    with cols[i % 5]:
        inputs[feature] = st.number_input(feature, value=np.random.randn())

if st.button("🔍 Analyze Traffic", type="primary"):
    # Demo prediction logic
    values = list(inputs.values())
    score = np.abs(np.mean(values))
    
    if score < 0.5:
        prediction = "Normal"
        confidence = 0.95
    elif score < 1.0:
        prediction = "Port Scan"
        confidence = 0.82
    elif score < 1.5:
        prediction = "DDoS"
        confidence = 0.78
    else:
        prediction = "Malware"
        confidence = 0.65
    
    # Display results
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Predicted Attack", prediction)
    
    with col2:
        st.metric("📊 Confidence", f"{confidence:.1%}")
    
    with col3:
        st.metric("🤖 Model", model)
    
    # Show probability distribution
    st.subheader("📈 Attack Probability Distribution")
    
    # Create fake probabilities
    probs = np.random.rand(len(attack_types))
    probs = probs / probs.sum()
    probs[np.where(np.array(attack_types) == prediction)[0][0]] = confidence
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(attack_types, probs)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Attack Type Probabilities')
    st.pyplot(fig)

# CSV Upload section
st.markdown("---")
st.header("📁 Or Upload CSV File")

uploaded_file = st.file_uploader("Upload network traffic data", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} records")
    st.dataframe(df.head())
    
    if st.button("🚀 Predict All"):
        # Simple demo predictions
        predictions = []
        for _ in range(len(df)):
            score = np.random.rand()
            if score > 0.8:
                predictions.append("DDoS")
            elif score > 0.6:
                predictions.append("Port Scan")
            elif score > 0.4:
                predictions.append("SQL Injection")
            elif score > 0.2:
                predictions.append("Malware")
            else:
                predictions.append("Normal")
        
        df['Prediction'] = predictions
        st.dataframe(df[['Prediction'] + list(df.columns[:3])].head(10))
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )
