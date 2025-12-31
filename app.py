# app.py - Simplified Cybersecurity Attack Classifier
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Cyber Attack Classifier",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Cybersecurity Attack Classification System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "XGBoost"]
    )
    
    st.markdown("---")
    st.header("📊 Data Input")
    
    input_mode = st.radio(
        "Choose input method:",
        ["Manual Entry", "CSV Upload"]
    )

# Main app logic
def load_demo_model():
    """Create a simple demo model if real ones aren't available"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Create a simple demo model
    np.random.seed(42)
    X_demo = np.random.randn(100, 10)
    y_demo = np.random.choice(['Normal', 'DDoS', 'PortScan'], 100)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_demo)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_demo)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y_encoded)
    
    return model, scaler, le, ['feature_'+str(i) for i in range(10)]

def main():
    # Try to load models
    try:
        if model_choice == "Random Forest":
            model = joblib.load('random_forest_model.pkl')
        elif model_choice == "Logistic Regression":
            model = joblib.load('logistic_regression_model.pkl')
        else:
            model = joblib.load('xgboost_model.pkl')
        
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        st.success("✅ Models loaded successfully!")
        
    except Exception as e:
        st.warning(f"⚠️ Using demo mode: {str(e)[:100]}")
        model, scaler, le, feature_names = load_demo_model()
    
    if input_mode == "Manual Entry":
        st.header("📝 Manual Data Entry")
        
        # Create input fields
        cols = st.columns(4)
        input_data = {}
        
        for i, feature in enumerate(feature_names[:12]):  # Show first 12 features
            col_idx = i % 4
            with cols[col_idx]:
                input_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    step=0.1,
                    key=feature
                )
        
        # Add default for other features
        for feature in feature_names[12:]:
            input_data[feature] = 0.0
        
        if st.button("🚀 Predict Attack", type="primary"):
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Scale features
            X_scaled = scaler.transform(input_df[feature_names])
            
            # Predict
            prediction = model.predict(X_scaled)
            prediction_label = le.inverse_transform(prediction)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Attack", prediction_label)
            
            with col2:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_scaled)[0]
                    confidence = np.max(probs)
                    st.metric("Confidence", f"{confidence:.2%}")
            
            with col3:
                st.metric("Model Used", model_choice)
            
            # Show probabilities if available
            if hasattr(model, 'predict_proba'):
                st.subheader("📈 Attack Probabilities")
                
                probs = model.predict_proba(X_scaled)[0]
                prob_df = pd.DataFrame({
                    'Attack Type': le.classes_,
                    'Probability': probs
                }).sort_values('Probability', ascending=False)
                
                # Display as bar chart
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(prob_df['Attack Type'][:5], prob_df['Probability'][:5])
                ax.set_xlabel('Probability')
                ax.set_title('Top 5 Attack Predictions')
                st.pyplot(fig)
    
    else:  # CSV Upload mode
        st.header("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload CSV with network traffic features"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
                
                with st.expander("📋 Preview Data"):
                    st.dataframe(df.head())
                
                if st.button("🔍 Analyze All Records", type="primary"):
                    # Ensure we have the right features
                    missing_features = set(feature_names) - set(df.columns)
                    if missing_features:
                        for feature in missing_features:
                            df[feature] = 0.0
                        st.warning(f"Added {len(missing_features)} missing features")
                    
                    # Scale and predict
                    X_scaled = scaler.transform(df[feature_names])
                    predictions = model.predict(X_scaled)
                    prediction_labels = le.inverse_transform(predictions)
                    
                    # Add predictions to dataframe
                    result_df = df.copy()
                    result_df['Predicted_Attack'] = prediction_labels
                    
                    if hasattr(model, 'predict_proba'):
                        confidences = np.max(model.predict_proba(X_scaled), axis=1)
                        result_df['Confidence'] = confidences
                    
                    # Show results
                    st.subheader("📊 Prediction Results")
                    
                    # Attack distribution
                    attack_counts = pd.Series(prediction_labels).value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Attack Distribution:**")
                        for attack, count in attack_counts.items():
                            st.write(f"{attack}: {count} records")
                    
                    with col2:
                        fig, ax = plt.subplots()
                        attack_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        ax.set_ylabel('')
                        st.pyplot(fig)
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="attack_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
