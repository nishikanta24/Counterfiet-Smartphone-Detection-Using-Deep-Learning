# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# --- Configuration ---
# IMPORTANT: This path MUST point to your saved model file.
# Ensure this file is accessible in the environment where you run Streamlit.
MODEL_PATH = "resnet50_best_single_model.pth"

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
@st.cache_resource # Cache the model and related info loading
def load_model_and_info(model_path):
    """Loads the checkpoint, extracts info, builds model, and loads state dict."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        st.error("Please ensure the path is correct and the file is accessible.")
        return None, None # Return None if model can't be loaded

    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # --- Extract info from checkpoint ---
        # Check if class_names and num_classes are in the checkpoint, otherwise define defaults
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            num_classes = len(class_names)
        else:
            # Fallback: You might need to define these manually if not in checkpoint
            st.warning("Class names not found in checkpoint. Using placeholders.")
            # Example: num_classes = 3 # Replace with your actual number
            # Example: class_names = ['Class0', 'Class1', 'Class2'] # Replace with your actual names
            # It's better to save them during training (as your script does)
            st.stop() # Stop if essential info is missing

        st.write(f"Model trained for {num_classes} classes: {class_names}")

        # --- Build Model Architecture ---
        # Load the ResNet50 base architecture (pre-trained weights not needed here)
        model = models.resnet50(weights=None) # Use weights=None

        # Get the number of input features for the classifier
        in_features = model.fc.in_features

        # Replace the final fully connected layer to match the loaded num_classes
        model.fc = nn.Linear(in_features, num_classes)

        # --- Load Trained Weights ---
        # Extract model state dict, handling potential variations in save format
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: # Common alternative key
            model_state = checkpoint['state_dict']
        else:
            # Assume the checkpoint itself is the state_dict (less common for dict saves)
            model_state = checkpoint
            st.warning("Checkpoint structure assumption: Loading entire checkpoint as state_dict.")

        # Handle potential 'module.' prefix if the model was saved using DataParallel
        # Create a new state_dict without the 'module.' prefix
        new_state_dict = {}
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


        model = model.to(device)
        model.eval() # Set model to evaluation mode

        st.success(f"Model loaded successfully from {model_path}")
        if 'epoch' in checkpoint:
            st.write(f"(Trained for {checkpoint['epoch']} epochs, Val Acc: {checkpoint.get('val_accuracy', 'N/A'):.4f})")

        return model, class_names
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None, None # Return None on error


# --- Image Transformations ---
# Use the same transformations as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet standard input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# --- Prediction Function ---
def predict(model, image, class_names):
    """Preprocesses the image and returns the predicted class name and confidence."""
    if model is None or class_names is None:
        st.error("Model not loaded correctly. Cannot predict.")
        return None, None

    try:
        # Ensure image is RGB
        image = image.convert("RGB")

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension and send to device

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1) # Get probabilities
            confidence, predicted_idx = torch.max(probabilities, 1) # Get highest probability and index

        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --- Streamlit User Interface ---
st.set_page_config(page_title="ResNet50 Image Classifier", layout="wide")
st.title("🖼️ Image Classification with ResNet50")
st.write(f"Using device: **{device}**")
st.write("Upload an image and the trained ResNet50 model will predict its class.")
st.markdown("---") # Separator

# Load the model and class names (cached)
model, class_names = load_model_and_info(MODEL_PATH)

# Only proceed if the model loaded successfully
if model and class_names:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with col2:
                st.write("### Prediction:")
                if st.button("🔍 Classify Image"):
                    with st.spinner('Classifying...'):
                        predicted_class, confidence_score = predict(model, image, class_names)

                    if predicted_class is not None:
                        st.success(f"Predicted Class: **{predicted_class}**")
                        st.info(f"Confidence: **{confidence_score:.2%}**")
                    else:
                        st.error("Prediction failed. Check logs for details.")

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
else:
    st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure the file exists and is valid.")
    st.stop() # Stop the app if model loading failed critically