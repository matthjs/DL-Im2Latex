from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch
from PIL import Image

# Load model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("Matthijs0/im2latex")
tokenizer = AutoTokenizer.from_pretrained("Matthijs0/im2latex")
feature_extractor = AutoFeatureExtractor.from_pretrained("Matthijs0/im2latex")

# Prepare an image
image = Image.open("./image.jpg").convert("RGB")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

# Generate LaTeX formula
generated_ids = model.generate(pixel_values)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Generated LaTeX formula:", generated_texts[0])
