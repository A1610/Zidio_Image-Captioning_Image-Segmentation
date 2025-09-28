from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 1) Model + processor load
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2) Image load
img_path = r"C:\Users\24mdsml004\Downloads\download.jpg"
raw_image = Image.open(img_path).convert("RGB")

# 3) Preprocess + generate caption
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)

caption = processor.decode(out[0], skip_special_tokens=True)

print(f"🖼️ Image: {img_path}")
print(f"📝 Caption: {caption}")
