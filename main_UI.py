import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Retention Dashboard", layout="wide")
st.title("🎯 Smart Customer Retention Insights")
st.markdown("""
Upload a CSV file with customer attributes, and we'll highlight customers at high risk of leaving.
Act fast and offer tailored deals to retain them! 🔁💼
""")

uploaded_file = st.file_uploader("📂 Upload Customer CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        # Match features to model
        model_features = model.feature_names_in_
        missing_features = [col for col in model_features if col not in df.columns]

        if missing_features:
            st.warning(f"Missing required features: {missing_features}")
        else:
            df_selected = df[model_features]
            df_scaled = pd.DataFrame(scaler.transform(df_selected), columns=model_features)
            
            predictions = model.predict(df_scaled)
            probabilities = model.predict_proba(df_scaled)[:, 1]

            df_result = df.copy()
            df_result['Churn Probability'] = probabilities
            df_result['Churn Risk'] = np.where(predictions == 1, "🚨 At Risk", "✅ Retained")

            st.subheader("📊 Prediction Summary")
            total = len(df_result)
            risk_count = np.sum(predictions)
            st.success(f"Out of {total} customers, **{risk_count}** are at risk of churning. Time to act!")

            st.subheader("🔍 Customers at Risk - Offer Special Deals")
            at_risk_customers = df_result[df_result['Churn Risk'] == "🚨 At Risk"]
            st.dataframe(at_risk_customers.reset_index(drop=True))

    except Exception as e:
        st.error(f"An error occurred: {e}")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load saved model and scaler
# model = joblib.load("xgboost_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # Define expected feature columns
# expected_columns = [
#     'Customer_Age', 'Gender', 'Dependent_count', 'Months_on_book',
#     'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
#     'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
#     'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
#     'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
#     'Education_Level_Doctorate', 'Education_Level_Graduate', 'Education_Level_High School',
#     'Education_Level_Post-Graduate', 'Education_Level_Uneducated', 'Education_Level_Unknown',
#     'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Unknown',
#     'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K',
#     'Income_Category_Less than $40K', 'Income_Category_Unknown',
#     'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver'
# ]

# # Streamlit UI
# st.title("🔍 Customer Churn Prediction App")
# st.markdown("Upload a CSV file with customer details to predict churn probability.")

# # File uploader
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.subheader("📄 Uploaded Data")
#         st.write(df.head())

#         # Step 1: Filter only expected columns
#         missing_cols = [col for col in expected_columns if col not in df.columns]
#         if missing_cols:
#             st.error(f"Missing required columns: {missing_cols}")
#         else:
#             input_df = df[expected_columns].copy()
#             input_df = input_df.astype('float32')

#             # Step 2: Normalize using saved scaler
#             input_scaled = pd.DataFrame(scaler.transform(input_df), columns=expected_columns)

#             # Step 3: Predict churn
#             predictions = model.predict(input_scaled)
#             probabilities = model.predict_proba(input_scaled)

#             # Step 4: Show results
#             df_result = df.copy()
#             df_result['Prediction'] = predictions
#             df_result['Churn Probability'] = probabilities[:, 1]

#             st.subheader("📊 Prediction Results")
#             st.write(df_result)

#             churn_count = np.sum(predictions)
#             st.success(f"Number of predicted churns: {churn_count} out of {len(predictions)}")

#     except Exception as e:
#         st.error(f"Prediction error: {e}")

