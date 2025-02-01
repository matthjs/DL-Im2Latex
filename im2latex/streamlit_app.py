"""
HOW TO RUN:  streamlit run streamlit_app.py
"""
import streamlit as st
import pyperclip
from PIL import Image, ImageGrab
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from im2latex.evaluators.swingradcam import swin_reshape_transform, SwinEncoderWrapper, GradCamAdaptor


# Load model, tokenizer, and feature extractor
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("Matthijs0/im2latex_base")
    tokenizer = AutoTokenizer.from_pretrained("Matthijs0/im2latex_base")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    return model, tokenizer, feature_extractor


model, tokenizer, feature_extractor = load_model()

# Streamlit App
st.title("Image to LaTeX Formula")

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

# GradCam visualization option
show_gradcam = st.checkbox("Show GradCam Visualization", value=False)
target_layers = [model.encoder.encoder.layers[-1].blocks[-1].layernorm_after]  # Last layer

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

                # GradCam visualization
                if show_gradcam:
                    with GradCamAdaptor(model=SwinEncoderWrapper(model.encoder, feature_extractor),
                                 target_layers=target_layers,
                                 reshape_transform=swin_reshape_transform) as cam:
                        grayscale_cam_batch = cam(input_tensor=pixel_values,
                                                  aug_smooth=True, eigen_smooth=True)
                        idx = 0
                        for grayscale_cam in grayscale_cam_batch:
                            # Convert to numpy and normalize to [0, 1]
                            # Apparently gradcam really wants this bruh
                            rgb_image = pixel_values[idx].cpu().numpy()
                            rgb_image = np.transpose(rgb_image, (1, 2, 0))
                            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
                            rgb_image = rgb_image.astype(np.float32)  # Ensure type is float32

                            visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
                            idx += 1

                        st.image(visualization, caption="GradCam Visualization", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
