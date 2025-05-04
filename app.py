import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the trained model and label encoder
model = joblib.load('rf_model_cicids2017.pkl')
label_encoder = joblib.load('label_encoder_cicids2017.pkl')

st.set_page_config(page_title="AI Intrusion Detection System", layout="wide")

st.title("🔐 AI-Powered Intrusion Detection System")
st.markdown("Upload a CSV file with network traffic to detect threats using a trained ML model.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and preprocess uploaded data
        df = pd.read_csv(uploaded_file)

        # Strip column names
        df.columns = df.columns.str.strip()

        # Clean data
        df = df.replace([float('inf'), float('-inf')], 0)
        df = df.fillna(0)

        # Check if 'Label' is in the uploaded data, and drop it if so
        if 'Label' in df.columns:
            df = df.drop('Label', axis=1)

        # Predict
        predictions = model.predict(df)
        prediction_labels = label_encoder.inverse_transform(predictions)

        # Display predictions
        df_results = df.copy()
        df_results['Prediction'] = prediction_labels

        st.success(f"✅ Prediction complete! Showing results:")
        st.dataframe(df_results)

        # Optional: Display threat summary
        threat_summary = df_results['Prediction'].value_counts()
        st.subheader("🔍 Threat Summary")
        st.bar_chart(threat_summary)

        # Optional: Highlight threats
        if 'BENIGN' in prediction_labels:
            st.info("Benign traffic detected.")
        else:
            st.error("⚠️ Potential threats detected!")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")
