import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔍 Customer Churn Prediction App")
st.markdown("Upload a CSV file with customer details to predict churn probability.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    def predict_churn(df):
        try:
            # Ensure the columns match the scaler expectations
            df = df[scaler.feature_names_in_]
            df = df.astype('float32')

            # Scale the data
            df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

            # Predict churn
            predictions = model.predict(df_scaled)
            probabilities = model.predict_proba(df_scaled)

            return predictions, probabilities

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

    if st.button("Predict Churn"):
        predictions, probabilities = predict_churn(df)

        if predictions is not None:
            df_result = df.copy()
            df_result['Prediction'] = predictions
            df_result['Churn Probability'] = probabilities[:, 1]

            st.subheader("📊 Prediction Results")
            st.write(df_result)

            # Show summary stats
            churn_count = np.sum(predictions)
            st.success(f"Number of predicted churns: {churn_count} out of {len(predictions)}")



#---------------------------------

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
