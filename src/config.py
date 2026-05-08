import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
ANALYSIS_DATA_DIR = os.path.join(DATA_DIR, "analysis")

# Ensure directories exist
for d in [RAW_DATA_DIR, CLEANED_DATA_DIR, ANALYSIS_DATA_DIR]:
    os.makedirs(d, exist_ok=True)
