# Remove zero-value no2_avg

import os
import csv
from tqdm import tqdm

# --- Configuration ---

# 1. Define Paths
OUTPUT_DIR = "/content/drive/MyDrive/hackhub"
INPUT_CSV = "/content/drive/MyDrive/hackhub/training_data_with_avg.csv"
FINAL_CSV = os.path.join(OUTPUT_DIR, "training_data_final.csv")

def clean_zero_value_entries():
  
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at '{INPUT_CSV}'")
        print("Please run the previous script to generate it first.")
        return

    print(f"Starting data cleaning process...")
    print(f"Reading data from: {INPUT_CSV}")

    kept_rows = []

    rows_processed = 0
    rows_deleted = 0

    with open(INPUT_CSV, 'r', newline='') as infile:
        reader = csv.reader(infile)
        
        header = next(reader)

        input_rows = list(reader)

    for row in tqdm(input_rows, desc="Scanning for zero-value entries"):
        rows_processed += 1

        if len(row) < 3:
            print(f"  - Warning: Skipping malformed row: {row}")
            continue

        sat_path_relative, no2_path_relative, avg_val_str = row

        
        if float(avg_val_str) == 0.0:
            rows_deleted += 1

           
            full_sat_path = os.path.join(OUTPUT_DIR, sat_path_relative)
            full_no2_path = os.path.join(OUTPUT_DIR, no2_path_relative)

            
            if os.path.exists(full_sat_path):
                os.remove(full_sat_path)
            else:
                print(f"  - Warning: Could not find sat image to delete: {full_sat_path}")

            # Delete the NO2 image
            if os.path.exists(full_no2_path):
                os.remove(full_no2_path)
            else:
                print(f"  - Warning: Could not find no2 image to delete: {full_no2_path}")

        else:
            # If the value is not 0, keep row
            kept_rows.append(row)

    print("\nWriting the cleaned data to a new CSV file...")

    with open(FINAL_CSV, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header) 
        writer.writerows(kept_rows) 

    rows_kept = len(kept_rows)
    print("\n--- Cleaning Complete! ---")
    print(f"Total rows processed: {rows_processed}")
    print(f"Rows and images deleted (value was 0.0): {rows_deleted}")
    print(f"Rows kept: {rows_kept}")
    print(f"Final clean dataset is saved at: '{FINAL_CSV}'")

# main
if __name__ == "__main__":
    clean_zero_value_entries()
