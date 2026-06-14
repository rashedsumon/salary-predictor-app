import streamlit as st
import pandas as pd
import numpy as np
from model import get_or_train_model
from data_loader import load_salary_data

# Set layout configurations
st.set_page_config(page_title="ML Engineer Salary Predictor", page_icon="💰", layout="centered")

st.title("💰 Machine Learning Salary Predictor")
st.markdown("Predict estimated annual salaries globally based on historical data collected throughout 2024.")

# Automatically download data and load the predictive model
@st.cache_resource
def initialize_app():
    # Load model (triggers training if it runs for the first time on Streamlit cloud)
    model = get_or_train_model()
    # Extract unique categories from raw data for dropdowns
    raw_df = load_salary_data()
    return model, raw_df

try:
    with st.spinner("Downloading dataset and building predictive intelligence..."):
        model, raw_df = initialize_app()
    
    st.success("App environment initialized successfully!")
    
    st.subheader("Enter Professional Profiles")
    
    # Creating selection options directly mapping columns of the raw dataset
    col1, col2 = st.columns(2)
    
    with col1:
        experience = st.selectbox("Experience Level", 
                                  options=raw_df['experience_level'].unique(),
                                  help="EN: Entry-level, MI: Mid-level, SE: Senior, EX: Executive")
        
        employment_type = st.selectbox("Employment Type", 
                                       options=raw_df['employment_type'].unique(),
                                       help="FT: Full-time, PT: Part-time, CT: Contract, FL: Freelance")
        
        # Filter matching standard titles from raw data or allow 'Other'
        top_titles = list(raw_df['job_title'].value_counts().index[:20]) + ['Other']
        job_title = st.selectbox("Job Title", options=top_titles)

    with col2:
        employee_residence = st.selectbox("Employee Residence Country", 
                                          options=sorted(raw_df['employee_residence'].unique()))
        
        company_location = st.selectbox("Company Base Country Location", 
                                        options=sorted(raw_df['company_location'].unique()))

    # Build evaluation row
    input_data = pd.DataFrame([{
        'experience_level': experience,
        'employment_type': employment_type,
        'job_title': job_title,
        'employee_residence': employee_residence,
        'company_location': company_location
    }])

    st.markdown("---")
    
    if st.button("Predict Estimated Salary", type="primary"):
        # Predict 
        prediction = model.predict(input_data)[0]
        
        # Display response outputs
        st.metric(label="Predicted Target Salary", value=f"${prediction:,.2f} USD")
        st.caption("Disclaimer: Outputs are generated estimations built directly over the public Kaggle 2024 market salary records.")

except Exception as e:
    st.error(f"An unexpected error occurred during application runtime initialization: {e}")