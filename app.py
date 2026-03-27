import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. SETUP AND CONFIGURATION
device = torch.device("cpu")

class_names = ['leo', 'not_leo'] 

# 2. MODEL LOADING FUNCTION
@st.cache_resource
def load_model():
    # Initialize the same architecture used in training (ResNet18)
    model = models.resnet18()
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load the trained weights from the saved PyTorch file
    # map_location=device ensures it works safely on CPU, even if trained on an M-chip or GPU
    model.load_state_dict(torch.load('leo_model.pth', map_location={'mps': 'cpu', 'mps:0': 'cpu'}, weights_only=True))
    
    # Set the model to evaluation mode (turns off dropout, batch normalization updates)
    model.eval()
    return model

# 3. IMAGE PREPROCESSING FUNCTION
def process_image(image):
    # These transformations MUST perfectly match the validation transforms 
    # used in the Jupyter Notebook so the model sees the data exactly how it expects.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add a batch dimension (B, C, H, W) -> (1, 3, 224, 224)
    input_tensor = preprocess(image).unsqueeze(0)
    return input_tensor

# 4. STREAMLIT USER INTERFACE
# Setup the UI headers
st.title("Spot the Leo: Image Classifier")
st.write("Upload a headshot, and my trained ResNet-18 model will determine if the photo is of Leonardo DiCaprio or someone else!")

# File uploader widget allowing standard image formats
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image using Pillow (PIL) and ensure it's in RGB format
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the uploaded image to the user
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    # Load the trained model
    model = load_model()
    
    # Preprocess the image for the model
    input_tensor = process_image(image)
    
    # ---------------------------------------------------------
    # 5. INFERENCE AND PREDICTION (Notice the indentation!)
    # ---------------------------------------------------------
    # Turn off gradient calculations for faster inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Apply softmax to convert raw outputs into percentages/probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get the predicted class index (0 or 1)
        _, predicted_idx = torch.max(output, 1)
        
    # Extract the string name and format the confidence percentage
    predicted_class = class_names[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item() * 100
    
    # Display results
    if predicted_class == 'leo':
        st.success(f"**Match found!** We're looking at Leonardo DiCaprio! (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"**No match.** This is NOT Leo. (Confidence: {confidence:.2f}%)")
        