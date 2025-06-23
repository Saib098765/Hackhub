# User Interface

!pip install -U gradio

import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import gradio as gr
import numpy as np

# Model Setup
MODEL_PATH = "/content/vgg16_no2_regressor.pth"

def get_vgg_for_regression():
    model = models.vgg16(weights=None)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_features, 1)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_vgg_for_regression()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction
def predict_no2(image: Image.Image):
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor).item()

    # Annotate the image
    annotated = image.resize((224, 224)).copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([0, 0, 224, 30], fill="white")
    draw.text((10, 5), f"Predicted NO₂: {output:.2f} ppm", fill="black")

    return annotated, f"{output:.2f}"

with gr.Blocks(theme=gr.themes.Base()) as demo:
    with gr.Row(variant="panel"):
        gr.Markdown("## 🛰️ **NO₂ Estimator from Satellite Images**")
        

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Satellite Image")
            predict_btn = gr.Button("Predict NO₂")
        with gr.Column():
            img_output = gr.Image(type="pil", label="NO₂ Output Image")
            val_output = gr.Textbox(label="Predicted NO₂ Value (ppm)", interactive=False)

    predict_btn.click(fn=predict_no2, inputs=[img_input], outputs=[img_output, val_output])

    gr.Markdown("### 📘 Project Description")
    gr.Markdown("""
**Tagline:**  
Leveraged a VGG-based Convolutional Neural Network to analyze air quality patterns across diverse terrains, enabling terrain-specific air quality estimation for any location in India using satellite and ground sensor data.

**Why This Matters:**  
Having precise air quality predictions is essential with growing emissions and harmful substances being released into the air.  
Current resources for air quality data typically provide information only for larger regions, with limited availability for smaller, localized areas.  
That’s why we came up with the idea of predicting NO₂ levels for specific terrain types in small, targeted regions.
""")

demo.launch()
