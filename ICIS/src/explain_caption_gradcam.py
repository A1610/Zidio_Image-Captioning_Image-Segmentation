import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------- CONFIG --------
CKPT = r"C:\Users\24mdsml004\Desktop\Arsh19\ICIS\checkpoints\blip_base_coco\best_epoch1"
IMG = r"C:\Users\24mdsml004\Desktop\Arsh19\datasets\coco\images\val2017\000000071226.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LOAD MODEL --------
processor = BlipProcessor.from_pretrained(CKPT, local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(CKPT, local_files_only=True).to(device)
model.eval()

# -------- IMAGE PREP --------
image = Image.open(IMG).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# -------- GENERATE CAPTION --------
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20)
caption = processor.decode(output[0], skip_special_tokens=True)
print("📝 Caption:", caption)

# -------- Tokenize caption to find important words --------
caption_tokens = processor.tokenizer.tokenize(caption)
print("🔑 Tokens:", caption_tokens)

# -------- Choose important words manually or via filter --------
important_words = [w for w in caption_tokens if w in ["dog", "book", "cat", "bed"]]
print("🎯 Important Words:", important_words)

# -------- TARGET LAYER (ViT encoder block last MLP) --------
target_layer = model.vision_model.encoder.layers[-1].mlp.fc2

# -------- reshape_transform for ViT (BLIP vision) --------
def blip_vit_reshape_transform(activations):
    cfg = model.vision_model.config
    H = cfg.image_size
    W = cfg.image_size if hasattr(cfg, "image_size") else H
    ps = cfg.patch_size
    x = activations[:, 1:, :]  
    h = H // ps
    w = W // ps
    x = x.reshape(x.size(0), h, w, x.size(-1))
    x = x.permute(0, 3, 1, 2).contiguous()
    return x

# -------- Wrap vision model --------
class VisionWrapper(torch.nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
    def forward(self, x):
        out = self.vision_model(x)
        return out.last_hidden_state

vision_wrapped = VisionWrapper(model.vision_model).to(device)

# -------- Token-target GradCAM --------
class TokenTarget:
    def __init__(self, token_idx):
        self.token_idx = token_idx
    def __call__(self, output):
        return output[:, self.token_idx, :].mean()

# -------- GradCAM --------
cam = GradCAM(
    model=vision_wrapped,
    target_layers=[target_layer],
    reshape_transform=blip_vit_reshape_transform
)

input_tensor = inputs["pixel_values"]

# -------- Loop over important words --------
for word in important_words:
    token_idx = caption_tokens.index(word) + 1  # +1 because [CLS] at start
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[TokenTarget(token_idx)],
        eigen_smooth=True
    )[0]

    # Resize CAM to original image size
    orig_w, orig_h = image.size
    grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))
    grayscale_cam_resized = cv2.GaussianBlur(grayscale_cam_resized, (11, 11), 0)

    # Overlay
    img_np = np.array(image, dtype=np.float32) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam_resized, use_rgb=True)

    # Show heatmap for this word
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_image)
    plt.title(f"Grad-CAM for '{word}' in: {caption}")
    plt.axis("off")
    plt.show()
