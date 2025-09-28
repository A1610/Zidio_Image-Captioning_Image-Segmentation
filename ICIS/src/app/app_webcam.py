# app/legend_app_live.py
import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

import torch
from torch.cuda.amp import autocast
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# ---------------- CONFIG - EDIT PATHS IF NEEDED ----------------
BLIP_CKPT = r"C:\Users\24mdsml004\Desktop\Arsh19\ICIS\checkpoints\blip_base_coco\best_epoch1"
YOLO_MODEL = "yolov8s-seg.pt"  # or yolov8n-seg.pt if nano only
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------

st.set_page_config(page_title="Legendary AI — Caption + Segmentation (Live)", layout="wide")
st.title("🖼️ Arsh — Image Caption + Segmentation (Upload + Live Webcam)")

# ---------------- Model loading (cached) ----------------
@st.cache_resource(show_spinner=True)
def load_caption_model(ckpt_path: str):
    processor = BlipProcessor.from_pretrained(ckpt_path, use_fast=True)  # ✅ use_fast=True added
    model = BlipForConditionalGeneration.from_pretrained(ckpt_path)
    model.eval()
    if DEVICE == "cuda":
        model.to(DEVICE)
    return processor, model

@st.cache_resource(show_spinner=True)
def load_seg_model(model_name: str):
    seg = YOLO(model_name)
    try:
        if DEVICE == "cuda":
            seg.model.to(DEVICE)
    except Exception:
        pass
    return seg

with st.spinner("Loading models (this can take some seconds)..."):
    processor, cap_model = load_caption_model(BLIP_CKPT)
    seg_model = load_seg_model(YOLO_MODEL)

# ---------------- util functions ----------------
def generate_caption(pil_img: Image.Image, max_new_tokens: int = 30) -> str:
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        if DEVICE == "cuda":
            with autocast():
                out = cap_model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            out = cap_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def run_segmentation_on_array(img_array: np.ndarray) -> np.ndarray:
    h, w = img_array.shape[:2]
    scale = IMG_SIZE / max(h, w) if max(h, w) > IMG_SIZE else 1.0
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_array, (new_w, new_h))
    else:
        resized = img_array.copy()

    results = seg_model(resized, device=0 if DEVICE == "cuda" else "cpu")
    out_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    for r in results:
        try:
            masks = r.masks.data if r.masks is not None else None
        except Exception:
            masks = None

        if masks is not None:
            for mask in masks:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (out_img.shape[1], out_img.shape[0]))  # ✅ fix
                color = np.array([0, 255, 0], dtype=np.uint8)
                alpha = 0.45
                out_img[mask_resized > 0.5] = (
                    out_img[mask_resized > 0.5] * (1 - alpha) + color * alpha
                ).astype(np.uint8)

        try:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = f"{seg_model.names[cls_id]} {score:.2f}"
                cv2.rectangle(out_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(
                    out_img, label, (xyxy[0], max(0, xyxy[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
                )
        except Exception:
            pass

    if scale != 1.0:
        out_img = cv2.resize(out_img, (w, h))

    return out_img

def pil_to_cv2_rgb(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------- Instance-aware captioning ----------------
def instance_captions_for_image(pil_img: Image.Image, seg_model, processor, cap_model,
                                device="cuda", max_instances=6, min_conf=0.35, img_size=640):
    import numpy as np
    import cv2
    from PIL import Image

    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    results = seg_model(orig_bgr, imgsz=img_size)
    overlay = cv2.cvtColor(orig_bgr.copy(), cv2.COLOR_BGR2RGB)
    instances = []

    for r in results:
        masks = None
        try:
            masks = r.masks.data if r.masks is not None else None
        except Exception:
            masks = None
        boxes = r.boxes if r.boxes is not None else []
        for idx, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < min_conf:
                continue
            cls_id = int(box.cls[0])
            label = seg_model.names[cls_id] if cls_id in seg_model.names else str(cls_id)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            # crop
            crop = orig_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            inputs = processor(images=pil_crop, return_tensors="pt").to(device)
            with torch.no_grad():
                if device == "cuda":
                    with autocast():
                        out = cap_model.generate(**inputs, max_new_tokens=30)
                else:
                    out = cap_model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # draw on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{label} ({conf:.2f}): {caption}"
            cv2.putText(overlay, text, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            instances.append({"label": label, "conf": conf, "caption": caption})

            if len(instances) >= max_instances:
                break
    return overlay, instances

# ---------------- UI: tabs for Upload vs Webcam ----------------
tab1, tab2 = st.tabs(["Upload Image", "Live Webcam"])

# ---------- Upload tab ----------
with tab1:
    st.markdown("**Upload an image** to get caption + segmentation.")
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil, caption="Original", use_container_width=True)
        with col2:
            st.subheader("Caption")
            caption = generate_caption(pil)
            st.success(caption)

            st.subheader("Segmentation Overlay")
            img_bgr = pil_to_cv2_rgb(pil)
            seg_out = run_segmentation_on_array(img_bgr)
            st.image(seg_out, channels="RGB", use_container_width=True)

        # ✅ New block: Instance-aware captions
        st.subheader("Segmentation + Instance Captions")
        overlay_rgb, instances = instance_captions_for_image(
            pil, seg_model, processor, cap_model,
            device=DEVICE, max_instances=6, min_conf=0.35, img_size=640
        )
        st.image(overlay_rgb, caption="Instance-aware overlay", use_container_width=True)
        for i, it in enumerate(instances, 1):
            st.markdown(f"**{i}. {it['label']}** (conf {it['conf']:.2f}) — _{it['caption']}_")

# ---------- Webcam tab ----------
with tab2:
    st.markdown("**Live webcam** — click Start, then Stop to end the stream.")
    cols = st.columns([1, 3])
    left, right = cols

    start = left.button("Start Webcam")
    stop = left.button("Stop Webcam")
    caption_mode = left.checkbox("Show captions on frames", value=True)
    seg_mode = left.checkbox("Show segmentation", value=True)
    framerate = left.slider("Max FPS (approx)", min_value=1, max_value=15, value=8)

    frame_placeholder = right.empty()

    if "cap" not in st.session_state:
        st.session_state.cap = None
        st.session_state.streaming = False

    if start:
        if st.session_state.cap is None:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for b in backends:
                cap = cv2.VideoCapture(0, b)
                if cap.isOpened():
                    st.session_state.cap = cap
                    st.session_state.streaming = True
                    time.sleep(0.5)
                    break

    if stop:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.streaming = False

    if st.session_state.cap is not None and st.session_state.streaming:
        try:
            last_time = 0
            while st.session_state.streaming:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.warning("Webcam frame not received.")
                    break

                now = time.time()
                if now - last_time < 1.0 / framerate:
                    time.sleep(0.01)
                    continue
                last_time = now

                frame = cv2.flip(frame, 1)
                display_img = frame.copy()

                if seg_mode:
                    try:
                        seg_img = run_segmentation_on_array(display_img)
                        display_rgb = seg_img
                    except Exception:
                        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                else:
                    display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

                caption_text = ""
                if caption_mode:
                    try:
                        pil_for = Image.fromarray(cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
                        caption_text = generate_caption(pil_for)
                    except Exception:
                        caption_text = "Caption error"

                if caption_mode and caption_text:
                    cv2.putText(
                        display_rgb, caption_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                    )

                frame_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

        except Exception as e:
            st.error(f"Webcam streaming stopped: {e}")
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
                st.session_state.streaming = False

    if st.session_state.cap is None:
        frame_placeholder.empty()
