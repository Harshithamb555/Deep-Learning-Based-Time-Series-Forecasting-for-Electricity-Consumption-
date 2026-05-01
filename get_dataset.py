# get_dataset.py
import os
import zipfile
import io
import requests
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
DATA_DIR = "data"
TXT_NAME = "household_power_consumption.txt"
CSV_NAME = "household_power_consumption.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Created folder:", DATA_DIR)

def download_and_extract_zip(url):
    print("Downloading dataset (this may take a few seconds)...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    if TXT_NAME in z.namelist():
        print("Extracting", TXT_NAME)
        z.extract(TXT_NAME, DATA_DIR)
        return os.path.join(DATA_DIR, TXT_NAME)
    else:
        raise RuntimeError("Expected file not found in zip")

def convert_txt_to_csv(txt_path, csv_path):
    print("Reading raw dataset (this may take a minute)...")
    # dataset uses semicolon delimiter and '?' for missing values
    df = pd.read_csv(
        txt_path,
        sep=';',
        header=0,
        na_values='?',
        low_memory=False
    )
    # Combine Date and Time into a single datetime column
    print("Parsing datetime and cleaning data...")
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    # Drop rows where datetime parsing failed
    df = df.dropna(subset=['datetime'])
    # Set index and drop original Date/Time columns
    df = df.set_index('datetime')
    df = df.drop(columns=['Date', 'Time'], errors='ignore')
    # Save to CSV (ISO datetime index)
    print("Saving CSV:", csv_path)
    df.to_csv(csv_path, index=True)
    print("Saved. Rows:", len(df))

def main():
    ensure_data_dir()
    txt_path = os.path.join(DATA_DIR, TXT_NAME)
    # If text file not already present, download+extract
    if not os.path.exists(txt_path):
        txt_path = download_and_extract_zip(URL)
    else:
        print("Found existing text file:", txt_path)
    # Convert to CSV path expected by your project
    if os.path.exists(CSV_PATH):
        print("CSV already exists at", CSV_PATH)
    else:
        convert_txt_to_csv(txt_path, CSV_PATH)
    print("Done. You can now run: python __main__.py")

if __name__ == "__main__":
    main()
