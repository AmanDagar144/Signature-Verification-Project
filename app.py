# ---------------------------------------------
# 🖥️ Signature Verification Streamlit App
# ---------------------------------------------
import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image

# Load trained model
model = joblib.load("signature_model.pkl")

# Preprocess uploaded image
def preprocess_image(img):
    st.write(f"Original image mode: {img.mode}")  

    # Convert to RGB if needed
    if img.mode == 'P':
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode not in ['RGB', 'L']:
        raise ValueError(f"Unsupported image mode: {img.mode}. Please upload RGB, RGBA, or Grayscale images.")

    img_np = np.array(img)
    st.write(f"Image shape after conversion: {img_np.shape}")  

    # Convert RGB to Grayscale
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif len(img_np.shape) == 2:
        gray = img_np
    else:
        raise ValueError("Unsupported image format after conversion.")

    # Resize, blur, threshold, normalize
    gray = cv2.resize(gray, (220, 155))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    gray = gray / 255.0  # Normalize

    return gray

# Extract HOG features
def extract_features(img):
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

# Streamlit UI
st.set_page_config(page_title="Signature Verification", layout="centered")
st.title("✍️ Signature Verification System")
st.write("Upload a signature image to check if it's **genuine or forged**.")

uploaded_file = st.file_uploader("Upload a signature image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Signature", use_container_width=True)

        # Preprocess image and extract features
        processed = preprocess_image(image)
        features = extract_features(processed).reshape(1, -1)

        # Predict with model
        prediction = model.predict(features)[0]
        label = "✅ Genuine" if prediction == 0 else "❌ Forged"

        st.markdown(f"### Prediction: {label}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
