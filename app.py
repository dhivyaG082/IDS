import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load model and encoder
model = joblib.load('models/rf_model_cicids2017.pkl')
label_encoder = joblib.load('models/label_encoder_cicids2017.pkl')

st.set_page_config(page_title="AI Intrusion Detection", layout="wide")
st.title("🔐 AI-Powered Intrusion Detection System")
st.markdown("Upload network traffic CSV to detect threats.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df.replace([float('inf'), float('-inf')], 0, inplace=True)
        df.fillna(0, inplace=True)

        if 'Label' in df.columns:
            df.drop(columns='Label', inplace=True)

        predictions = model.predict(df)
        labels = label_encoder.inverse_transform(predictions)

        df['Prediction'] = labels
        st.success("✅ Prediction Complete!")
        st.dataframe(df)

        summary = df['Prediction'].value_counts()
        st.subheader("🔍 Threat Summary")
        st.bar_chart(summary)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please upload a CSV file.")
