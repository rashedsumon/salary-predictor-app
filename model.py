import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

from data_loader import load_salary_data

MODEL_FILE = "salary_pipeline.pkl"

def build_and_train_model():
    """
    Loads data, isolates high-cardinality values, constructs 
    a preprocessing and modeling pipeline, trains it, and caches it.
    """
    # 1. Load Data
    df = load_salary_data()
    
    # Target column is 'salary_in_usd'
    target_col = 'salary_in_usd'
    
    # Selecting the primary informative features for salary estimation
    features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location']
    
    X = df[features]
    y = df[target_col]
    
    # Quick data cleaning: Filter rare job titles to prevent pipeline breakdown
    top_titles = X['job_title'].value_counts().index Harvey = X['job_title'].value_counts().index[:20]
    X['job_title'] = X['job_title'].apply(lambda x: x if x in top_titles else 'Other')

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Create Preprocessing Pipeline (handling categorical text data)
    categorical_features = features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # 4. Define the complete unified pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # 5. Train Model
    print("Training the Random Forest model...")
    model_pipeline.fit(X_train, y_train)
    
    # 6. Export/Save Model Object
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"Model successfully saved to {MODEL_FILE}")
    
    return model_pipeline

def get_or_train_model():
    """
    Returns the trained pipeline model. If it already exists locally, 
    loads it immediately; otherwise, boots up the training process first.
    """
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return build_and_train_model()

if __name__ == "__main__":
    build_and_train_model()