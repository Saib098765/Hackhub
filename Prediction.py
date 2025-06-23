# Prediction on a New Image

import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np


# Paths
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/hackhub'
MODEL_PATH = 'vgg16_no2_regressor.pth'
IMAGE_TO_PREDICT_PATH = '/content/drive/MyDrive/hackhub/b.png'

def get_vgg_for_regression():
    model = models.vgg16(weights=None) # We don't need pretrained weights, we're loading our own
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_features, 1)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the trained model
print(f"Loading model from: {MODEL_PATH}")
model = get_vgg_for_regression()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    
    model.eval()
except FileNotFoundError:
    print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
    print("Please make sure the path is correct and you have run the training cell.")
else:
    # 2. Define the image transformations (identical to training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Process the image
    print(f"Loading and processing image: {IMAGE_TO_PREDICT_PATH}")
    try:
        image = Image.open(IMAGE_TO_PREDICT_PATH).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension and send to device

        # 4. Make prediction
        print("Predicting...")
        with torch.no_grad(): # We don't need to calculate gradients for prediction
            prediction = model(image_tensor)

        predicted_value = prediction.item() # Extract the single value from the tensor

        print("\n✅ --- Prediction Result --- ✅")
        print(f"The predicted average NO2 value for the image is: {predicted_value:.4f}")

    except FileNotFoundError:
        print(f"ERROR: The image file was not found at '{IMAGE_TO_PREDICT_PATH}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")
