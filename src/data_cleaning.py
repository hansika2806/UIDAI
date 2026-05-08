import os
import glob
import zipfile
import pandas as pd
import numpy as np
from config import RAW_DATA_DIR, CLEANED_DATA_DIR

STATE_MAPPING = {
    "Andaman & Nicobar Islands": "Andaman And Nicobar Islands",
    "Jammu & Kashmir": "Jammu And Kashmir",
    "Dadra & Nagar Haveli": "Dadra And Nagar Haveli",
    "Daman & Diu": "Daman And Diu",
    "The Dadra And Nagar Haveli And Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Orissa": "Odisha",
    "Pondicherry": "Puducherry",
    "West  Bengal": "West Bengal",
    "West Bangal": "West Bengal",
    "Westbengal": "West Bengal",
    "West Bengli": "West Bengal",
    "Chhatisgarh": "Chhattisgarh",
    "Uttaranchal": "Uttarakhand"
}

MERGE_UT_MAPPING = {
    "Dadra And Nagar Haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu"
}

REQUIRED_RAW_PATTERNS = {
    "enrolment": ["api_data_aadhar_enrolment*.zip", "api_data_aadhar_enrolment*.csv"],
    "demographic": ["api_data_aadhar_demographic*.zip", "api_data_aadhar_demographic*.csv"],
    "biometric": ["api_data_aadhar_biometric*.zip", "api_data_aadhar_biometric*.csv"],
}


def validate_raw_inputs():
    missing = []
    for dataset_type, patterns in REQUIRED_RAW_PATTERNS.items():
        found = False
        for pattern in patterns:
            if glob.glob(os.path.join(RAW_DATA_DIR, pattern)):
                found = True
                break
        if not found:
            missing.append(dataset_type)
    if missing:
        missing_lines = "\n".join(
            f"- {dataset_type}: place matching zip/csv files in {RAW_DATA_DIR}"
            for dataset_type in missing
        )
        raise FileNotFoundError(
            "Required raw UIDAI datasets are missing.\n"
            f"{missing_lines}"
        )

def extract_zips():
    print("Extracting any zip files in raw data directory...")
    zips = glob.glob(os.path.join(RAW_DATA_DIR, "*.zip"))
    for zip_path in zips:
        extract_folder = zip_path.replace('.zip', '')
        if not os.path.exists(extract_folder):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_folder)
        else:
            print(f"Folder {extract_folder} already exists, skipping extraction.")

def load_and_clean_enrolment():
    print("Processing Enrolment Data...")
    # Try to find CSVs in subdirectories or directly in RAW_DATA_DIR
    files = glob.glob(os.path.join(RAW_DATA_DIR, "api_data_aadhar_enrolment*", "**", "*.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(RAW_DATA_DIR, "api_data_aadhar_enrolment*.csv"))
        
    if not files:
        print("Warning: No enrolment CSV files found.")
        return None

    df_list = []
    for f in files:
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df['state'] = df['state'].str.strip().str.title()
        df['district'] = df['district'].str.strip().str.title()
        df['pincode'] = df['pincode'].astype(str)
        
        # Drop negative counts
        df = df[(df['age_0_5'] >= 0) & (df['age_5_17'] >= 0) & (df['age_18_greater'] >= 0)]
        df_list.append(df)

    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)
    
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['date', 'state', 'district', 'pincode', 'age_0_5', 'age_5_17', 'age_18_greater'])
    
    # State mapping
    df['state'] = df['state'].replace('100000', np.nan)
    df = df.dropna(subset=['state'])
    df['state'] = df['state'].replace(STATE_MAPPING).replace(MERGE_UT_MAPPING)
    
    output_path = os.path.join(CLEANED_DATA_DIR, "aadhaar_enrolment_clean_final.csv")
    df.to_csv(output_path, index=False)
    print(f"Enrolment data saved to {output_path}")
    return df

def load_and_clean_update_data(dataset_type, valid_states):
    print(f"Processing {dataset_type.capitalize()} Data...")
    
    folder_pattern = f"api_data_aadhar_{dataset_type}*"
    files = glob.glob(os.path.join(RAW_DATA_DIR, folder_pattern, "**", "*.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(RAW_DATA_DIR, f"api_data_aadhar_{dataset_type}*.csv"))

    if not files:
        print(f"Warning: No {dataset_type} CSV files found.")
        return None

    df_list = []
    for f in files:
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df['state'] = df['state'].str.strip().str.title()
        df['district'] = df['district'].str.strip().str.title()
        df['pincode'] = df['pincode'].astype(str)
        
        if dataset_type == 'demographic':
            df = df[(df['demo_age_5_17'] >= 0) & (df['demo_age_17_'] >= 0)]
        else:
            df = df[(df['bio_age_5_17'] >= 0) & (df['bio_age_17_'] >= 0)]
            
        df_list.append(df)

    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)
    
    # Drop duplicates
    if dataset_type == 'demographic':
        subset = ['date', 'state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_']
    else:
        subset = ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_']
    df = df.drop_duplicates(subset=subset)

    # State mapping
    df['state'] = df['state'].replace('100000', np.nan)
    df['state'] = df['state'].replace(STATE_MAPPING).replace(MERGE_UT_MAPPING)
    
    # Filter by valid states
    if valid_states is not None:
        df = df[df['state'].isin(valid_states)].copy()
        
    output_path = os.path.join(CLEANED_DATA_DIR, f"aadhaar_{dataset_type}_update_clean_final.csv")
    df.to_csv(output_path, index=False)
    print(f"{dataset_type.capitalize()} data saved to {output_path}")
    return df

def main():
    print("Starting Data Cleaning Pipeline...")
    validate_raw_inputs()
    extract_zips()
    
    enrolment_df = load_and_clean_enrolment()
    
    valid_states = None
    if enrolment_df is not None:
        valid_states = set(enrolment_df['state'].unique())
        
    load_and_clean_update_data('demographic', valid_states)
    load_and_clean_update_data('biometric', valid_states)
    print("Data Cleaning Pipeline Complete ✅")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc)
