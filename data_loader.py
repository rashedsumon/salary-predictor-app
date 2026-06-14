import os
import glob
import pandas as pd
import kagglehub

def load_salary_data():
    """
    Downloads the latest version of the target salary dataset 
    via kagglehub and loads the core CSV into a pandas DataFrame.
    """
    print("Checking/Downloading dataset via kagglehub...")
    # Downloads the dataset files to a local cache directory
    download_path = kagglehub.dataset_download("chopper53/machine-learning-engineer-salary-in-2024")
    
    # Locate the CSV file inside the downloaded directory
    csv_files = glob.glob(os.path.join(download_path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded path: {download_path}")
        
    # Read the first matching dataset file (typically salaries.csv)
    df = pd.read_csv(csv_files[0])
    return df

if __name__ == "__main__":
    # Test script execution
    data = load_salary_data()
    print("Successfully loaded dataset columns:", data.columns.tolist())
    print(data.head(2))