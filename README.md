
LoyaltyLens - Customer Churn Prediction

LoyaltyLens is an advanced machine learning project designed to predict customer churn in financial institutions. It combines Exploratory Data Analysis (EDA), multiple machine learning models, SMOTE balancing, hyperparameter tuning, model interpretation with LIME, and real-time prediction deployment through a Streamlit web application.

Project Structure
EDA_and_Model_Training.ipynb - Full EDA and model training code.
xgboost_model.pkl - Trained best model (XGBoost).
scaler.pkl - Scaler used for data normalization.
streamlit_app.py - Streamlit frontend app for churn prediction.

How to Run

1. Requirements:
   - Python 3.8 or above
   - Recommended IDE: VSCode, PyCharm, or Jupyter Notebook
   - Install required libraries:

   pip install -r requirements.txt
   
2. Install Streamlit:

   pip install streamlit

3. Run the Streamlit App:

   streamlit run streamlit_app.py
   
4. Usage:
   - Upload customer CSV file through the web interface.
   - Get churn predictions instantly.

Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Streamlit
- LIME (Local Interpretable Model-agnostic Explanations)

Note
Make sure that the xgboost_model.pkl and scaler.pkl files are in the same directory as the streamlit_app.py file for correct loading of the model.
