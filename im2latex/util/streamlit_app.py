"""
HOW TO RUN:  streamlit run streamlit_app.py
"""
import streamlit as st
import pyperclip
import torch
from PIL import Image, ImageGrab
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor


# Load model, tokenizer, and feature extractor
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("Matthijs0/im2latex_base")
    tokenizer = AutoTokenizer.from_pretrained("Matthijs0/im2latex_base")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    return model, tokenizer, feature_extractor


model, tokenizer, feature_extractor = load_model()

# Streamlit App
st.title("Screenshot to LaTeX Formula")

# Screenshot section
if st.button("Take Screenshot"):
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    st.image(screenshot, caption="Captured Screenshot", use_container_width=True)


# Image upload section
uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
else:
    image = None
    if st.button("Use Last Screenshot"):
        try:
            image = Image.open("screenshot.png").convert("RGB")
        except FileNotFoundError:
            st.error("No screenshot found. Please take a screenshot first.")

# Process image
if image:
    if st.button("Process to LaTeX"):
        with st.spinner("Processing..."):
            try:
                # Extract pixel values
                pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

                # Generate LaTeX formula
                generated_ids = model.generate(pixel_values)
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # Display result
                latex_formula = generated_texts[0]
                st.success("LaTeX Formula Generated:")
                st.code(latex_formula, language="latex")

                # Copy to clipboard
                pyperclip.copy(latex_formula)
                st.info("LaTeX formula copied to clipboard!")

            except Exception as e:
                st.error(f"Error: {str(e)}")