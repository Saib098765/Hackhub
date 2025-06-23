# SPLIT THE DATA INTO 100 IMAGES

pip install Pillow pandas

import cv2
import numpy as np
import os
import csv


# 1. Define Paths
INPUT_DIR = "/content/drive/MyDrive/hackhub/input_images"
OUTPUT_DIR = "/content/drive/MyDrive/hackhub"
CSV_FILENAME = os.path.join(OUTPUT_DIR, "training_data.csv")

# 2. Patch Generation Parameters(We will create a 10x10 grid to get 100 patches per image.)
GRID_SIZE = 10

# 3. Quality Control Parameters
# Reject a patch if it's more than 95% percentage black.
BLACK_THRESHOLD = 0.95

# 4. Function for Padding
def pad_image_to_size(image, target_h, target_w):
    h, w = image.shape[:2]

    # Calculate padding needed for height and width
    pad_h = target_h - h
    pad_w = target_w - w

    # Ensure no negative padding
    if pad_h < 0 or pad_w < 0:
        # This case should not happen if logic is correct, but as a safeguard
        print("  - Warning: Image is larger than target. Cropping is not implemented. Returning original.")
        return image

    # Distribute padding to top/bottom and left/right to center the image
    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad

    # Define the border color as white
    # (255, 255, 255) for color images, 255 for grayscale
    border_color = [255, 255, 255] if len(image.shape) == 3 else 255

    # Apply padding
    padded_image = cv2.copyMakeBorder(
        image,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=border_color
    )

    return padded_image

def create_directories():
    """Creates the necessary output directories if they don't exist."""
    os.makedirs(os.path.join(OUTPUT_DIR, "sat_patches"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "no2_patches"), exist_ok=True)
    print(f"Output directories created/ensured in '{OUTPUT_DIR}'")

def is_patch_valid(patch, threshold):
    """
    Checks if a patch is valid by ensuring it's not mostly black.
    This check is performed on the satellite image patch.
    """
    if patch is None or patch.size == 0:
        return False

    if np.max(patch) == 0:
        return False

    if threshold < 1.0:
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) > 2 else patch
        black_pixels = np.sum(gray_patch == 0)
        total_pixels = gray_patch.size
        if (black_pixels / total_pixels) >= threshold:
            return False

    return True

def split_and_save_images():
    """
    Main function to load image pairs, pad them to equal size, split into patches,
    perform quality control, save them, and create a CSV manifest.
    """
    create_directories()

    csv_data = []
    total_valid_patches = 0

    files = os.listdir(INPUT_DIR)
    prefixes = sorted(list(set([f.split('-')[0] for f in files if '-' in f])))

    print(f"Found image prefixes: {prefixes}\n")

    for prefix in prefixes:
        # Construct file names dynamically
        sat_path, no2_path = None, None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            temp_sat_path = os.path.join(INPUT_DIR, f"{prefix}-sat{ext}")
            temp_no2_path = os.path.join(INPUT_DIR, f"{prefix}-no2{ext}")
            if os.path.exists(temp_sat_path): sat_path = temp_sat_path
            if os.path.exists(temp_no2_path): no2_path = temp_no2_path

        if not sat_path or not no2_path:
            print(f"--- WARNING: Skipping prefix '{prefix}'. Could not find both satellite and NO2 images. ---")
            continue

        print(f"--- Processing image pair for prefix: {prefix} ---")

        # Load images
        sat_image = cv2.imread(sat_path, cv2.IMREAD_COLOR)
        no2_image = cv2.imread(no2_path, cv2.IMREAD_COLOR)

        if sat_image is None or no2_image is None:
            print(f"  ERROR: Could not read one of the images for prefix '{prefix}'. Skipping.")
            continue

        # match the no2 and satellite image sizes(make them equal to the larger image size)
        h_sat, w_sat = sat_image.shape[:2]
        h_no2, w_no2 = no2_image.shape[:2]

        if (h_sat, w_sat) != (h_no2, w_no2):
            print(f"  - WARNING: Image sizes do not match. Padding to a common size.")
            print(f"    Sat image: {h_sat}x{w_sat}, NO2 image: {h_no2}x{w_no2}")

            # Determine the target size (the max of both dimensions)
            target_h = max(h_sat, h_no2)
            target_w = max(w_sat, w_no2)
            print(f"    Padding both images to: {target_h}x{target_w}")

            # Pad images to the target size
            sat_image = pad_image_to_size(sat_image, target_h, target_w)
            no2_image = pad_image_to_size(no2_image, target_h, target_w)

        
        
        img_h, img_w = sat_image.shape[:2]
        print(f"  Processing with unified dimensions (H, W): ({img_h}, {img_w})")

        patch_h = img_h // GRID_SIZE
        patch_w = img_w // GRID_SIZE

        if patch_h == 0 or patch_w == 0:
            print(f"  ERROR: Image size ({img_h}x{img_w}) is too small to split into a {GRID_SIZE}x{GRID_SIZE} grid. Skipping.")
            continue

        print(f"  Calculated patch dimensions (H, W): ({patch_h}, {patch_w})")

        valid_patches_for_this_image = 0

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                y_start, x_start = i * patch_h, j * patch_w
                y_end, x_end = y_start + patch_h, x_start + patch_w

                sat_patch = sat_image[y_start:y_end, x_start:x_end]
                no2_patch = no2_image[y_start:y_end, x_start:x_end]

                if is_patch_valid(sat_patch, BLACK_THRESHOLD):
                    base_filename = f"{prefix}_patch_{i}_{j}.png"
                    sat_patch_filename = os.path.join("sat_patches", f"{prefix}_sat_{base_filename}")
                    no2_patch_filename = os.path.join("no2_patches", f"{prefix}_no2_{base_filename}")

                    full_sat_path = os.path.join(OUTPUT_DIR, sat_patch_filename)
                    full_no2_path = os.path.join(OUTPUT_DIR, no2_patch_filename)

                    cv2.imwrite(full_sat_path, sat_patch)
                    cv2.imwrite(full_no2_path, no2_patch)

                    csv_data.append([sat_patch_filename, no2_patch_filename])
                    valid_patches_for_this_image += 1

        print(f"  Saved {valid_patches_for_this_image} valid patches out of a possible {GRID_SIZE*GRID_SIZE}.")
        total_valid_patches += valid_patches_for_this_image

    if not csv_data:
        print("\nNo valid patches were generated. CSV file will not be created.")
        return

    print(f"\nTotal valid patches generated: {total_valid_patches}")

    header = ['satellite_image_path', 'no2_image_path']
    with open(CSV_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)

    print(f"Successfully created CSV manifest at: '{CSV_FILENAME}'")

# Main
if __name__ == "__main__":
    split_and_save_images()
