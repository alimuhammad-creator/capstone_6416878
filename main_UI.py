import streamlit as st
import joblib

# Load the model or scaler
model = joblib.load('xgboost_model.pkl')     # or 'rf_model.pkl', etc.
scaler = joblib.load('scaler.pkl')

# Show the expected columns
st.write("Columns expected by model:", model.feature_names_in_)
st.write("Columns expected by scaler:", scaler.feature_names_in_)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import MinMaxScaler

# # --- Load Trained Model and Scaler ---
# model = joblib.load("xgboost_model.pkl")  # Replace with your model path
# scaler = joblib.load("scaler.pkl")        # Replace with your scaler path

# # --- UI Configuration ---
# st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
# st.title("🧠 Customer Churn Prediction App")

# # --- Sidebar Navigation ---
# mode = st.sidebar.radio("Select Mode", ["Upload CSV", "Manual Entry", "Chatbot (Coming Soon)"])

# # --- Helper Function to Make Prediction ---
# def predict_churn(df):
#     df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
#     predictions = model.predict(df_scaled)
#     probabilities = model.predict_proba(df_scaled)[:, 1]
#     return predictions, probabilities

# # --- Phase 1: CSV Upload ---
# if mode == "Upload CSV":
#     st.header("📂 Upload a CSV File")
#     file = st.file_uploader("Upload your CSV file", type=["csv"])
#     if file:
#         data = pd.read_csv(file)
#         st.subheader("📄 Uploaded Data")
#         st.dataframe(data.head())

#         if st.button("Predict Churn"):
#             predictions, probabilities = predict_churn(data)
#             results = data.copy()
#             results['Prediction'] = np.where(predictions == 1, "Churn", "Stay")
#             results['Probability'] = np.round(probabilities, 2)

#             st.subheader("🔍 Results")
#             st.dataframe(results)

# # --- Phase 2: Manual Entry ---
# elif mode == "Manual Entry":
#     st.header("✍️ Enter Customer Details")

#     age = st.slider("Customer Age", 18, 90, 45)
#     credit_limit = st.number_input("Credit Limit", min_value=100.0, max_value=100000.0, value=5000.0)
#     dependents = st.slider("Dependent Count", 0, 5, 2)
#     gender = st.selectbox("Gender", ["Female", "Male"])

#     # Extend to more features as needed...

#     input_dict = {
#         "Customer_Age": age,
#         "Credit_Limit": credit_limit,
#         "Dependent_count": dependents,
#         "Gender": 0 if gender == "Female" else 1
#     }

#     input_df = pd.DataFrame([input_dict])

#     if st.button("Predict Customer Churn"):
#         pred, prob = predict_churn(input_df)
#         st.success(f"Prediction: {'Churn' if pred[0] == 1 else 'Stay'} (Confidence: {prob[0]*100:.1f}%)")

# # --- Phase 3: Chatbot Placeholder ---
# elif mode == "Chatbot (Coming Soon)":
#     st.header("💬 AI Assistant (Gemini/GPT Integration)")
#     st.info("This feature is under development. Soon, you'll be able to chat with an AI to assess churn risk based on customer info!")
