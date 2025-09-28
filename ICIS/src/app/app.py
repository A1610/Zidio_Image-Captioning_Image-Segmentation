import os
import torch
import cv2
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import numpy as np

# ---------- CONFIG ----------
# apna checkpoint ka absolute path yaha do
CKPT = r"C:\Users\24mdsml004\Desktop\Arsh19\ICIS\checkpoints\blip_base_coco\best_epoch1"
YOLO_MODEL = "yolov8n-seg.pt"  # or "runs/segment/train/weights/best.pt" if fine-tuned

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained(CKPT, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(CKPT, local_files_only=True).to(device)
    model.eval()
    return processor, model

@st.cache_resource
def load_seg_model():
    model = YOLO(YOLO_MODEL)
    return model

processor, cap_model = load_caption_model()
seg_model = load_seg_model()

# ---------- FUNCTIONS ----------
def generate_caption(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = cap_model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

def run_segmentation(image_path: str):
    results = seg_model(image_path, save=False)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    for r in results:
        masks = r.masks
        boxes = r.boxes
        if masks is not None:
            for mask in masks.data:
                mask = mask.cpu().numpy()

                # 🛠 FIX: resize mask to match image dimensions
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                # overlay green where mask == 1
                img[mask > 0.5] = (0, 255, 0)

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = seg_model.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(img, label, (xyxy[0], xyxy[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Legendary AI: Caption + Segmentation", layout="wide")
st.title("🖼️ ARSH : Image Caption + Segmentation")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_pil, caption="Original", use_container_width=True)

    with col2:
        st.subheader("Caption")
        caption = generate_caption(img_pil)
        st.success(caption)

    with col3:
        st.subheader("Segmentation")
        tmp_path = "temp.jpg"
        img_pil.save(tmp_path)
        seg_img = run_segmentation(tmp_path)
        st.image(seg_img, caption="Segmented", use_container_width=True)
