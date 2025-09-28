from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Path to your saved checkpoint
checkpoint_path = r"C:\Users\24mdsml004\Desktop\Arsh19\ICIS\checkpoints\blip_base_coco\best_epoch1"

# Load processor + model from local checkpoint
processor = BlipProcessor.from_pretrained(checkpoint_path)
model = BlipForConditionalGeneration.from_pretrained(checkpoint_path)

# Test image path
img_path = r"C:\Users\24mdsml004\Desktop\Arsh19\ICIS\src\runs\segment\predict\images.jpg"
raw_image = Image.open(img_path).convert("RGB")

# Generate caption
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("📝 Caption:", caption)
