# Avg n02 value

pip install tqdm

import pandas as pd
import os
import cv2
import numpy as np
from tqdm.notebook import tqdm

# Paths
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/hackhub'
OUTPUT_DIR = '/content/drive/MyDrive/hackhub'
INPUT_CSV = '/content/drive/MyDrive/hackhub/training_data.csv'
OUTPUT_CSV_WITH_AVG = '/content/drive/MyDrive/hackhub/training_data_with_avg.csv'

# Map the No2 pixels from NO2 map to the 15 classes from London Air map
COLOR_TO_VALUE_MAP_REFINED = {
    (120, 102, 79): 15.0,   (164, 84, 68): 17.5,    (212, 121, 68): 20.5,
    (222, 166, 82): 23.5,   (212, 200, 102): 26.5,  (191, 222, 127): 29.5,
    (184, 237, 179): 32.5,  (158, 230, 194): 35.5,  (130, 224, 212): 38.5,
    (181, 240, 255): 41.5,  (140, 224, 255): 44.5,  (155, 201, 255): 47.5,
    (158, 179, 255): 50.5,  (145, 145, 247): 53.5,  (153, 112, 194): 56.5,
    (130, 84, 138): 59.0
}
WHITE_PIXEL_BGR = np.array([255, 255, 255])


def calculate_average_no2_selective(patch):
    # Calculates the average NO2 value by ONLY considering pixels that map to a value of 15 or greater.
    
    color_keys = np.array(list(COLOR_TO_VALUE_MAP_REFINED.keys()), dtype=np.uint8)
    color_values = np.array(list(COLOR_TO_VALUE_MAP_REFINED.values()), dtype=np.float32)

   
    pixels = patch.reshape(-1, 3)

   
    distances = np.sum((color_keys - pixels[:, np.newaxis])**2, axis=2)
    closest_color_indices = np.argmin(distances, axis=1)

    value_map = color_values[closest_color_indices]
    valid_pixels_mask = value_map >= 15.0

    valid_values = value_map[valid_pixels_mask]

    # Calculate the average.
    if valid_values.size > 0:
        average = np.mean(valid_values)
    else:
        average = 0.0

    return average

print("--- Calculating average NO2 values (Selective >=15 method) ---")
with open(INPUT_CSV, 'r') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    input_rows = list(reader)

new_csv_data = []
for row in tqdm(input_rows, desc="Processing patches"):
    no2_patch_image = cv2.imread(os.path.join(OUTPUT_DIR, row[1]))
    if no2_patch_image is None: continue
    # Use the new SELECTIVE calculation function
    avg_value = calculate_average_no2_selective(no2_patch_image)
    new_csv_data.append([row[0], row[1], f"{avg_value:.4f}"])

with open(OUTPUT_CSV_WITH_AVG, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['satellite_image_path', 'no2_image_path', 'no2_avg_val'])
    writer.writerows(new_csv_data)
print("Finished calculating selective averages.")

# Clean Zero-Value Entries (delete no2_values, satellite images from dataset which yield 0 avg no2 value)

FINAL_CSV = os.path.join(OUTPUT_DIR, "training_data_final.csv")

print("\n--- Cleaning zero-value entries ---")
df = pd.read_csv(OUTPUT_CSV_WITH_AVG)
df_to_keep = df[df['no2_avg_val'] > 0.0]
df_to_delete = df[df['no2_avg_val'] <= 0.0]

print(f"Keeping {len(df_to_keep)} patches. Removing {len(df_to_delete)} patches.")

if not df_to_delete.empty:
    print("Deleting image files for zero-value patches...")
    for index, row in tqdm(df_to_delete.iterrows(), total=df_to_delete.shape[0], desc="Deleting files"):
        sat_path = os.path.join(OUTPUT_DIR, row['satellite_image_path'])
        no2_path = os.path.join(OUTPUT_DIR, row['no2_image_path'])
        if os.path.exists(sat_path): os.remove(sat_path)
        if os.path.exists(no2_path): os.remove(no2_path)

df_to_keep.to_csv(FINAL_CSV, index=False)

print(f"\n✅ Final, clean data saved to '{FINAL_CSV}'")
