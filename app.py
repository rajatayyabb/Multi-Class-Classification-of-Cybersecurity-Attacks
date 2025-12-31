# app.py - Cybersecurity Attack Classifier (No Dependencies)
import streamlit as st
import random

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
        ["Random Forest", "Logistic Regression", "XGBoost", "Ensemble"]
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    st.markdown("---")
    if st.button("🔄 Load Demo Data", use_container_width=True):
        st.session_state.demo_loaded = True
        st.success("Demo data loaded!")

# Initialize session state
if 'demo_loaded' not in st.session_state:
    st.session_state.demo_loaded = False

# Feature names (fixed set)
feature_names = [
    "Duration", "Protocol_Type", "Service", "Flag", 
    "Src_Bytes", "Dst_Bytes", "Land", "Wrong_Fragment",
    "Urgent", "Hot", "Num_Failed_Logins", "Logged_In",
    "Num_Compromised", "Root_Shell", "Su_Attempted",
    "Num_Root", "Num_File_Creations", "Num_Shells"
]

# Attack types
attack_types = [
    "Normal", "DoS", "Probe", "R2L", "U2R",
    "DDoS", "Port Scan", "SQL Injection", "Brute Force"
]

def predict_attack(input_data, model_name):
    """Simple prediction logic without ML libraries"""
    # Create a deterministic "prediction" based on inputs
    total_value = sum(abs(v) for v in input_data.values() if isinstance(v, (int, float)))
    
    # Simple logic for demo
    if total_value < 10:
        attack = "Normal"
        confidence = 0.95
    elif total_value < 20:
        attack = "Port Scan"
        confidence = 0.82
    elif total_value < 30:
        attack = "DoS"
        confidence = 0.75
    elif total_value < 40:
        attack = "DDoS"
        confidence = 0.68
    else:
        attack = "SQL Injection"
        confidence = 0.60
    
    # Add some randomness based on model choice
    if model_name == "Random Forest":
        confidence = min(confidence + 0.05, 0.99)
    elif model_name == "XGBoost":
        confidence = min(confidence + 0.03, 0.98)
    
    return attack, confidence

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Upload", "📊 Results"])

with tab1:
    st.header("Single Instance Prediction")
    
    # Create input fields in columns
    cols = st.columns(4)
    input_data = {}
    
    for i, feature in enumerate(feature_names):
        col_idx = i % 4
        with cols[col_idx]:
            input_data[feature] = st.number_input(
                feature,
                value=0.0 if i % 3 != 0 else 1.0,
                step=0.1,
                key=f"input_{feature}"
            )
    
    if st.button("🚀 Predict Attack", type="primary", use_container_width=True):
        attack, confidence = predict_attack(input_data, model_choice)
        
        # Display results
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Predicted Attack", attack)
        
        with col2:
            st.metric("📊 Confidence", f"{confidence:.1%}")
            if confidence < confidence_threshold:
                st.warning(f"Below threshold ({confidence_threshold:.0%})")
        
        with col3:
            st.metric("🤖 Model Used", model_choice)
        
        # Show all attack probabilities
        st.subheader("📈 Attack Probability Distribution")
        
        # Create fake probabilities
        prob_data = {atk: random.uniform(0, 0.3) for atk in attack_types}
        prob_data[attack] = confidence
        
        # Normalize
        total = sum(prob_data.values())
        for atk in prob_data:
            prob_data[atk] /= total
        
        # Sort and display
        sorted_probs = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)
        
        for atk, prob in sorted_probs[:5]:
            col_prog, col_text = st.columns([3, 1])
            with col_prog:
                st.progress(float(prob))
            with col_text:
                st.write(f"**{atk}**: {prob:.1%}")

with tab2:
    st.header("Batch CSV Processing")
    
    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=['csv'],
        help="Upload CSV with network features"
    )
    
    if uploaded_file is not None:
        # Simulate reading CSV
        st.success("✅ File uploaded successfully!")
        
        # Create sample data
        import pandas as pd
        import io
        
        # Read the file
        content = uploaded_file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        
        st.write(f"**Records loaded**: {len(df):,}")
        st.write(f"**Features found**: {len(df.columns)}")
        
        if st.button("🔍 Process All Records", type="primary"):
            # Simulate predictions
            predictions = []
            confidences = []
            
            for _ in range(min(100, len(df))):  # Limit to 100 for demo
                attack, confidence = predict_attack(
                    {f: random.random() for f in feature_names[:5]},
                    model_choice
                )
                predictions.append(attack)
                confidences.append(confidence)
            
            # Create results
            results_df = df.copy()
            results_df['Prediction'] = predictions[:len(results_df)]
            results_df['Confidence'] = confidences[:len(results_df)]
            
            # Show summary
            st.subheader("📊 Prediction Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                attack_counts = pd.Series(predictions).value_counts()
                st.write("**Attack Distribution:**")
                for atk, count in attack_counts.items():
                    st.write(f"{atk}: {count} records")
            
            with col2:
                # Simple chart using streamlit native
                chart_data = pd.DataFrame({
                    'Attack': attack_counts.index.tolist(),
                    'Count': attack_counts.values.tolist()
                })
                st.bar_chart(chart_data.set_index('Attack'))
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="attack_predictions.csv",
                mime="text/csv"
            )

with tab3:
    st.header("Model Performance")
    
    # Create demo performance data
    performance_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'XGBoost', 'Ensemble'],
        'Accuracy': [0.962, 0.923, 0.971, 0.978],
        'Precision': [0.958, 0.917, 0.967, 0.975],
        'Recall': [0.953, 0.912, 0.964, 0.973],
        'F1-Score': [0.955, 0.914, 0.965, 0.974]
    }
    
    # Display metrics
    st.subheader("📈 Performance Metrics")
    
    cols = st.columns(4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        with cols[idx]:
            max_val = max(p[metric] for p in [dict(zip(performance_data.keys(), values)) 
                          for values in zip(*performance_data.values())])
            st.metric(f"Best {metric}", f"{max_val:.3f}")
    
    # Show table
    import pandas as pd
    df_perf = pd.DataFrame(performance_data)
    st.dataframe(df_perf.style.format({
        'Accuracy': '{:.3f}',
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1-Score': '{:.3f}'
    }), use_container_width=True)
    
    # Confusion matrix demo
    st.subheader("🎯 Confusion Matrix (Sample)")
    
    # Create a simple confusion matrix
    cm_data = {
        'Actual \\ Predicted': attack_types[:5],
        **{atk: [random.randint(0, 50) for _ in range(5)] for atk in attack_types[:5]}
    }
    
    cm_df = pd.DataFrame(cm_data)
    st.dataframe(cm_df.set_index('Actual \\ Predicted'), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### 🔧 About This Demo
This is a **functional demo** of a cybersecurity attack classification system. 
In a production environment, this would connect to trained ML models.

**Features demonstrated:**
- Real-time attack prediction
- Multiple "model" comparison  
- Batch processing simulation
- Performance metrics display
- Results export functionality

*Note: This version uses simulated data and predictions for demonstration purposes.*
""")
