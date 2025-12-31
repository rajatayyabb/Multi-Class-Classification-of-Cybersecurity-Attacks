# app.py - Cybersecurity Attack Classification Web App
# Streamlit deployment version
# Based on Kaggle notebook by Tayyab Ali (2530-4007)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Cybersecurity Attack Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #F59E0B;
    }
    .attack-card {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        models = {}
        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl',
            'XGBoost': 'xgboost_model.pkl'
        }
        
        for name, file in model_files.items():
            models[name] = joblib.load(file)
        
        # Load preprocessing objects
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_encoders = joblib.load('feature_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        results_summary = joblib.load('results_summary.pkl')
        
        return {
            'models': models,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_encoders': feature_encoders,
            'feature_names': feature_names,
            'results_summary': results_summary
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def preprocess_input(df, feature_encoders, scaler):
    """Preprocess input data similar to training pipeline"""
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Encode categorical features
    for col in categorical_cols:
        if col in feature_encoders:
            # Handle unseen categories by encoding as -1
            known_categories = set(feature_encoders[col].classes_)
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in known_categories else 'unknown'
            )
            df_processed[col] = feature_encoders[col].transform(df_processed[col])
    
    # Scale features
    X_scaled = scaler.transform(df_processed)
    
    return X_scaled

def predict_attack(models, X_preprocessed, label_encoder):
    """Make predictions using all models"""
    predictions = {}
    
    for model_name, model in models.items():
        try:
            y_pred = model.predict(X_preprocessed)
            y_pred_labels = label_encoder.inverse_transform(y_pred)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_preprocessed)
                predictions[model_name] = {
                    'labels': y_pred_labels,
                    'probabilities': probabilities,
                    'encoded': y_pred
                }
            else:
                predictions[model_name] = {
                    'labels': y_pred_labels,
                    'probabilities': None,
                    'encoded': y_pred
                }
        except Exception as e:
            st.warning(f"Error with {model_name}: {str(e)}")
            continue
    
    return predictions

def create_confusion_matrix(y_true, y_pred, model_name):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=label_encoder.classes_[:len(np.unique(y_true))],
        y=label_encoder.classes_[:len(np.unique(y_true))],
        color_continuous_scale='Blues',
        title=f'{model_name} - Confusion Matrix'
    )
    fig.update_layout(width=700, height=600)
    return fig

def plot_attack_distribution(predictions_dict):
    """Plot attack type distribution from predictions"""
    all_predictions = []
    for model_name, pred_data in predictions_dict.items():
        all_predictions.extend(pred_data['labels'])
    
    df_counts = pd.DataFrame({'Attack Type': all_predictions})
    count_series = df_counts['Attack Type'].value_counts().reset_index()
    count_series.columns = ['Attack Type', 'Count']
    
    fig = px.bar(
        count_series,
        x='Attack Type',
        y='Count',
        color='Attack Type',
        title='Attack Type Distribution Across All Models',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        xaxis_title="Attack Type",
        yaxis_title="Count",
        showlegend=False,
        width=800,
        height=500
    )
    return fig

def get_model_performance_metrics():
    """Get performance metrics from saved results"""
    results = joblib.load('results_summary.pkl')
    return results['model_performance']

# Main app
def main():
    # Header
    st.markdown("<h1 class='main-header'>🛡️ Cybersecurity Attack Classification System</h1>", unsafe_allow_html=True)
    st.markdown("### Student: Tayyab Ali (2530-4007) | Department of Cyber Security")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=100)
        st.markdown("### 🎯 Navigation")
        
        app_mode = st.selectbox(
            "Choose Mode",
            ["🏠 Home", "📊 Model Performance", "🔮 Predict Attacks", "📁 Batch Prediction", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression", "XGBoost", "Ensemble (All Models)"]
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence score for predictions"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Model Status")
        
        # Load models
        if st.button("🔄 Load Models", use_container_width=True):
            with st.spinner("Loading models and preprocessing objects..."):
                loaded_data = load_models()
                if loaded_data:
                    st.session_state.models_data = loaded_data
                    st.session_state.model_loaded = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.error("❌ Failed to load models")
        
        if st.session_state.get('model_loaded', False):
            st.success("Models: ✅ Loaded")
            st.info(f"Features: {len(st.session_state.models_data['feature_names'])}")
            st.info(f"Attack Types: {len(st.session_state.models_data['label_encoder'].classes_)}")
        else:
            st.warning("Models: ⚠️ Not Loaded")
        
        st.markdown("---")
        st.markdown("### 📥 Quick Actions")
        
        # Sample data download
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(5),
            'feature2': np.random.randn(5),
            'feature_categorical': ['A', 'B', 'A', 'C', 'B']
        })
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_cyber_data.csv",
            mime="text/csv",
            help="Download a sample CSV file with correct format"
        )

    # Main content area
    if app_mode == "🏠 Home":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Project Overview")
            st.markdown("""
            This web application provides a multi-class classification system for identifying 
            various types of cybersecurity attacks using network traffic data.
            
            **Key Features:**
            - 🛡️ **Real-time Attack Prediction**: Classify network traffic as normal or specific attack types
            - 📊 **Multiple ML Models**: Compare Random Forest, Logistic Regression, and XGBoost
            - 📈 **Performance Visualization**: Detailed metrics and confusion matrices
            - 📁 **Batch Processing**: Upload CSV files for bulk prediction
            - 🔍 **Explainability**: Confidence scores and detailed predictions
            """)
            
            st.markdown("### 🎯 Supported Attack Types")
            if st.session_state.get('model_loaded', False):
                attack_types = st.session_state.models_data['label_encoder'].classes_
                for attack in attack_types:
                    st.markdown(f"<div class='attack-card'>• {attack}</div>", unsafe_allow_html=True)
            else:
                st.info("Load models to see supported attack types")
        
        with col2:
            st.markdown("### ⚡ Quick Start")
            st.markdown("""
            1. **Load Models** using the sidebar button
            2. **Choose Prediction Mode**:
               - Single prediction with manual input
               - Batch prediction with CSV upload
            3. **Select Model** from the sidebar
            4. **View Results** with detailed analysis
            """)
            
            st.markdown("### 📋 Dataset Info")
            if st.session_state.get('model_loaded', False):
                results_summary = st.session_state.models_data['results_summary']
                st.metric("Total Samples", f"{results_summary['n_samples_total']:,}")
                st.metric("Features", results_summary['n_features'])
                st.metric("Attack Types", len(results_summary['target_classes']))
    
    elif app_mode == "📊 Model Performance":
        st.markdown("<h2 class='sub-header'>📊 Model Performance Analysis</h2>", unsafe_allow_html=True)
        
        if not st.session_state.get('model_loaded', False):
            st.warning("⚠️ Please load models first from the sidebar")
            return
        
        # Load performance metrics
        performance_data = get_model_performance_metrics()
        df_performance = pd.DataFrame(performance_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_model = df_performance.loc[df_performance['F1-Score'].idxmax()]
            st.metric("Best Model", best_model['Model'])
        
        with col2:
            st.metric("Best F1-Score", f"{best_model['F1-Score']:.3f}")
        
        with col3:
            st.metric("Best Accuracy", f"{best_model['Accuracy']:.3f}")
        
        # Performance metrics table
        st.markdown("### 📈 Detailed Metrics")
        st.dataframe(
            df_performance.style.format({
                'Accuracy': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'Train Acc': '{:.3f}'
            }).background_gradient(cmap='Blues', subset=['Accuracy', 'F1-Score']),
            use_container_width=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart comparison
            fig = go.Figure(data=[
                go.Bar(name='Accuracy', x=df_performance['Model'], y=df_performance['Accuracy']),
                go.Bar(name='F1-Score', x=df_performance['Model'], y=df_performance['F1-Score']),
                go.Bar(name='Precision', x=df_performance['Model'], y=df_performance['Precision']),
                go.Bar(name='Recall', x=df_performance['Model'], y=df_performance['Recall'])
            ])
            fig.update_layout(
                title='Model Performance Comparison',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Train vs Test Accuracy
            fig = go.Figure(data=[
                go.Bar(name='Train Accuracy', x=df_performance['Model'], y=df_performance['Train Acc']),
                go.Bar(name='Test Accuracy', x=df_performance['Model'], y=df_performance['Accuracy'])
            ])
            fig.update_layout(
                title='Train vs Test Accuracy',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.markdown("### 🎯 Confusion Matrices")
        st.info("Confusion matrices show model performance on test data")
        
        # Note: In production, you might want to load pre-computed confusion matrices
        # For now, we'll show a placeholder or calculate if test data is available
    
    elif app_mode == "🔮 Predict Attacks":
        st.markdown("<h2 class='sub-header'>🔮 Single Instance Prediction</h2>", unsafe_allow_html=True)
        
        if not st.session_state.get('model_loaded', False):
            st.warning("⚠️ Please load models first from the sidebar")
            return
        
        models_data = st.session_state.models_data
        
        # Create input form based on feature names
        st.markdown("### 📝 Enter Feature Values")
        
        # Simple input for demo - in real scenario, you'd map all features
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        feature_names = models_data['feature_names']
        
        # Display first 9 features for input (adjust as needed)
        sample_features = feature_names[:9]
        
        for i, feature in enumerate(sample_features):
            col_idx = i % 3
            if col_idx == 0:
                with col1:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.1,
                        key=f"input_{feature}"
                    )
            elif col_idx == 1:
                with col2:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.1,
                        key=f"input_{feature}"
                    )
            else:
                with col3:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.1,
                        key=f"input_{feature}"
                    )
        
        # Add default values for remaining features
        for feature in feature_names[9:]:
            input_data[feature] = 0.0
        
        if st.button("🚀 Predict Attack Type", type="primary", use_container_width=True):
            with st.spinner("Analyzing network traffic..."):
                # Convert input to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Preprocess input
                X_preprocessed = preprocess_input(
                    input_df,
                    models_data['feature_encoders'],
                    models_data['scaler']
                )
                
                # Make predictions
                predictions = predict_attack(
                    models_data['models'],
                    X_preprocessed,
                    models_data['label_encoder']
                )
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                if selected_model == "Ensemble (All Models)":
                    # Show results from all models
                    for model_name, pred_data in predictions.items():
                        attack_type = pred_data['labels'][0]
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if pred_data['probabilities'] is not None:
                                confidence = np.max(pred_data['probabilities'][0])
                                st.metric(f"{model_name}", attack_type)
                                st.metric("Confidence", f"{confidence:.2%}")
                        
                        with col2:
                            if pred_data['probabilities'] is not None:
                                # Show probability distribution
                                prob_df = pd.DataFrame({
                                    'Attack Type': models_data['label_encoder'].classes_,
                                    'Probability': pred_data['probabilities'][0]
                                })
                                prob_df = prob_df.sort_values('Probability', ascending=False)
                                
                                fig = px.bar(
                                    prob_df.head(5),
                                    x='Attack Type',
                                    y='Probability',
                                    color='Probability',
                                    color_continuous_scale='RdYlGn_r'
                                )
                                fig.update_layout(
                                    title=f"{model_name} - Top 5 Predictions",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    # Show results for selected model
                    if selected_model in predictions:
                        pred_data = predictions[selected_model]
                        attack_type = pred_data['labels'][0]
                        
                        st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**Predicted Attack:**")
                            st.markdown(f"# {attack_type}")
                        
                        with col2:
                            if pred_data['probabilities'] is not None:
                                confidence = np.max(pred_data['probabilities'][0])
                                st.markdown(f"**Confidence:**")
                                st.markdown(f"# {confidence:.2%}")
                                
                                if confidence < confidence_threshold:
                                    st.warning(f"Low confidence (below {confidence_threshold:.0%} threshold)")
                        
                        with col3:
                            st.markdown(f"**Model Used:**")
                            st.markdown(f"# {selected_model}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Show detailed probabilities
                        if pred_data['probabilities'] is not None:
                            st.markdown("#### 📈 Detailed Probability Distribution")
                            
                            prob_df = pd.DataFrame({
                                'Attack Type': models_data['label_encoder'].classes_,
                                'Probability': pred_data['probabilities'][0]
                            })
                            prob_df = prob_df.sort_values('Probability', ascending=False)
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                fig = px.bar(
                                    prob_df,
                                    x='Attack Type',
                                    y='Probability',
                                    color='Probability',
                                    color_continuous_scale='RdYlGn_r',
                                    title="Probability Distribution Across All Attack Types"
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("#### 🏆 Top Predictions")
                                for idx, row in prob_df.head().iterrows():
                                    st.markdown(f"**{row['Attack Type']}**")
                                    st.progress(float(row['Probability']))
                                    st.markdown(f"`{row['Probability']:.2%}`")
                                    st.markdown("---")
    
    elif app_mode == "📁 Batch Prediction":
        st.markdown("<h2 class='sub-header'>📁 Batch Prediction from CSV</h2>", unsafe_allow_html=True)
        
        if not st.session_state.get('model_loaded', False):
            st.warning("⚠️ Please load models first from the sidebar")
            return
        
        models_data = st.session_state.models_data
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with network traffic data",
            type=['csv'],
            help="Upload a CSV file with the same features used during training"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df_uploaded = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df_uploaded
                
                st.success(f"✅ File loaded successfully! ({len(df_uploaded)} rows, {len(df_uploaded.columns)} columns)")
                
                # Show preview
                with st.expander("📋 Preview Uploaded Data"):
                    st.dataframe(df_uploaded.head(), use_container_width=True)
                
                # Check if columns match
                expected_features = set(models_data['feature_names'])
                uploaded_features = set(df_uploaded.columns)
                
                if expected_features.issubset(uploaded_features):
                    st.success("✅ All required features found in uploaded data")
                    
                    if st.button("🔍 Analyze All Records", type="primary", use_container_width=True):
                        with st.spinner("Processing batch data..."):
                            # Preprocess data
                            X_preprocessed = preprocess_input(
                                df_uploaded[models_data['feature_names']],
                                models_data['feature_encoders'],
                                models_data['scaler']
                            )
                            
                            # Make predictions
                            predictions = predict_attack(
                                models_data['models'],
                                X_preprocessed,
                                models_data['label_encoder']
                            )
                            
                            # Create results DataFrame
                            results_df = df_uploaded.copy()
                            
                            for model_name, pred_data in predictions.items():
                                results_df[f'{model_name}_Prediction'] = pred_data['labels']
                                if pred_data['probabilities'] is not None:
                                    max_probs = np.max(pred_data['probabilities'], axis=1)
                                    results_df[f'{modelName}_Confidence'] = max_probs
                            
                            # Display results
                            st.markdown("### 📊 Batch Prediction Results")
                            
                            # Summary statistics
                            if selected_model in predictions:
                                attack_counts = pd.Series(predictions[selected_model]['labels']).value_counts()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### 📈 Attack Distribution")
                                    fig = px.pie(
                                        values=attack_counts.values,
                                        names=attack_counts.index,
                                        title=f"Attack Type Distribution ({selected_model})",
                                        color_discrete_sequence=px.colors.qualitative.Set3
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown("#### 📋 Count Summary")
                                    for attack_type, count in attack_counts.items():
                                        st.markdown(f"**{attack_type}**: {count} records")
                            
                            # Show results table
                            with st.expander("📄 View Detailed Results", expanded=False):
                                st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv_results = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv_results.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="cyber_attack_predictions.csv">📥 Download Predictions CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                else:
                    missing_features = expected_features - uploaded_features
                    st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    st.info("Please ensure your CSV contains all required features")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    elif app_mode == "ℹ️ About":
        st.markdown("<h2 class='sub-header'>ℹ️ About This Project</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            
            This Cybersecurity Attack Classification System is designed to identify and classify 
            various types of network attacks using machine learning algorithms trained on network 
            traffic data.
            
            ### 🛠️ Technical Details
            
            **Models Implemented:**
            - **Random Forest**: Ensemble of decision trees for robust classification
            - **Logistic Regression**: Linear model with regularization
            - **XGBoost**: Gradient boosting for high accuracy
            
            **Preprocessing Pipeline:**
            1. Missing value imputation
            2. Categorical feature encoding
            3. Feature scaling
            4. Class imbalance handling
            
            **Performance Metrics:**
            - Accuracy, Precision, Recall, F1-Score
            - Confusion matrices
            - ROC curves (where applicable)
            
            ### 📚 Dataset Information
            
            The models were trained on a comprehensive cybersecurity dataset containing:
            - Multiple attack types (DDoS, Port Scan, SQL Injection, etc.)
            - Network traffic features (packet size, protocol type, duration, etc.)
            - Both normal and malicious traffic patterns
            
            ### 👨‍💻 Developer Information
            
            **Student**: Tayyab Ali (2530-4007)  
            **Department**: Cyber Security  
            **Project**: Multi-Class Classification of Cybersecurity Attacks
            
            ### 🚀 Deployment
            
            This application is deployed using:
            - **Streamlit**: Web application framework
            - **Scikit-learn**: Machine learning library
            - **Plotly**: Interactive visualizations
            - **Joblib**: Model serialization
            """)
        
        with col2:
            st.markdown("### 📈 Model Architecture")
            st.image("https://raw.githubusercontent.com/plotly/datasets/master/network.png", use_column_width=True)
            
            st.markdown("### 🔒 Security Features")
            st.markdown("""
            <div class='success-box'>
            <strong>Secure Processing</strong><br>
            • No data persistence<br>
            • Local processing only<br>
            • Encrypted transmissions
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Model Statistics")
            if st.session_state.get('model_loaded', False):
                results_summary = st.session_state.models_data['results_summary']
                
                metrics = [
                    ("Total Samples", f"{results_summary['n_samples_total']:,}"),
                    ("Training Samples", f"{results_summary['n_samples_train']:,}"),
                    ("Testing Samples", f"{results_summary['n_samples_test']:,}"),
                    ("Features", results_summary['n_features']),
                    ("Attack Types", len(results_summary['target_classes']))
                ]
                
                for label, value in metrics:
                    st.metric(label, value)

if __name__ == "__main__":
    main()