import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected feature columns
expected_columns = [
    'Customer_Age', 'Gender', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
    'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Education_Level_Doctorate', 'Education_Level_Graduate', 'Education_Level_High School',
    'Education_Level_Post-Graduate', 'Education_Level_Uneducated', 'Education_Level_Unknown',
    'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Unknown',
    'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K',
    'Income_Category_Less than $40K', 'Income_Category_Unknown',
    'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver'
]

# Streamlit UI
st.title("🔍 Customer Churn Prediction App")
st.markdown("Upload a CSV file with customer details to predict churn probability.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.write(df.head())

        # Step 1: Filter only expected columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            input_df = df[expected_columns].copy()
            input_df = input_df.astype('float32')

            # Step 2: Normalize using saved scaler
            input_scaled = pd.DataFrame(scaler.transform(input_df), columns=expected_columns)

            # Step 3: Predict churn
            predictions = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)

            # Step 4: Show results
            df_result = df.copy()
            df_result['Prediction'] = predictions
            df_result['Churn Probability'] = probabilities[:, 1]

            st.subheader("📊 Prediction Results")
            st.write(df_result)

            churn_count = np.sum(predictions)
            st.success(f"Number of predicted churns: {churn_count} out of {len(predictions)}")

    except Exception as e:
        st.error(f"Prediction error: {e}")



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
