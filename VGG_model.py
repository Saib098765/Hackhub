# VGG


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Hyperparameters
class Config:
    CSV_PATH = '/content/drive/MyDrive/hackhub/training_data_final.csv'
    IMAGE_ROOT_DIR = '/content/drive/MyDrive/hackhub'

    
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4  
    NUM_EPOCHS = 20       
    TEST_SPLIT_SIZE = 0.2 
    RANDOM_STATE = 42     


class NO2Dataset(Dataset):
    # Adding the Custom Dataset for loading satellite images and their NO2 values.

    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['satellite_image_path'])
        image = Image.open(img_path).convert('RGB')
        no2_value = float(row['no2_avg_val'])
        if self.transform:
            image = self.transform(image)
        target = torch.tensor([no2_value], dtype=torch.float32)

        return image, target

# Define model
def get_vgg_for_regression(pretrained=True):
    # Load a pre-trained VGG16 model and adapts its classifier for regression.

    # Load VGG16 with pre-trained weights from ImageNet
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)

    # Freeze the convolutional layers
    if pretrained:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the final layer for regression
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 1) # Output is a single value

    return model

if __name__ == "__main__":
  
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and Split Data
    df = pd.read_csv(config.CSV_PATH)

    # Split dataframe into training and validation sets
    train_df, val_df = train_test_split(
        df,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_STATE
    )
    print(f"Data split: {len(train_df)} training samples, {len(val_df)} validation samples.")

    # Define Image Transformations (VGG models expect 224x224 images and specific normalization.)
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NO2Dataset(train_df, config.IMAGE_ROOT_DIR, transform=data_transforms)
    val_dataset = NO2Dataset(val_df, config.IMAGE_ROOT_DIR, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = get_vgg_for_regression().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.LEARNING_RATE)

    # Training and Validation
    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        
        model.train() 
        running_train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Training]"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation Phase
        model.eval() 
        running_val_loss = 0.0

        with torch.no_grad(): 
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Validation]"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> "
              f"Train Loss: {epoch_train_loss:.4f} (RMSE: {np.sqrt(epoch_train_loss):.4f}) | "
              f"Val Loss: {epoch_val_loss:.4f} (RMSE: {np.sqrt(epoch_val_loss):.4f})")

    print("\n--- Training Finished ---")

    # Save state of model
    model_save_path = "vgg16_no2_regressor.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
