# ---------------- Instance-aware captioning utilities ----------------
import torch
import numpy as np
import cv2
from PIL import Image

def mask_to_bbox(mask):
    """Given binary mask (H,W), return bbox (x1,y1,x2,y2) or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)

def crop_instance_from_image(orig_bgr: np.ndarray, mask: np.ndarray, bbox=None, pad=8):
    """
    Create a crop for the instance:
    - Apply mask to keep instance and set background white
    - Crop to bbox with small pad
    Returns a PIL Image (RGB) suitable for captioner.
    """
    h, w = mask.shape
    bin_mask = (mask > 0.5).astype(np.uint8)

    if bbox is None:
        bbox = mask_to_bbox(bin_mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)

    crop_img = orig_bgr[y1:y2+1, x1:x2+1].copy()  # BGR
    crop_mask = bin_mask[y1:y2+1, x1:x2+1]

    # convert background to white where mask is zero
    if crop_img.ndim == 3:
        white = np.ones_like(crop_img, dtype=np.uint8) * 255
        crop_img = np.where(crop_mask[..., None] == 1, crop_img, white)

    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)

def instance_captions_for_image(
    pil_img: Image.Image,
    seg_model,
    processor,
    cap_model,
    device="cuda",
    max_instances=6,
    min_conf=0.3,
    img_size=640,
    max_new_tokens=30
):
    """
    Run YOLO segmentation + BLIP captioning on each detected instance.

    Args:
        pil_img (PIL.Image): input RGB image
        seg_model (YOLO): YOLOv8 segmentation model
        processor (BlipProcessor): BLIP processor
        cap_model (BlipForConditionalGeneration): BLIP caption model
        device (str): "cuda" or "cpu"
        max_instances (int): max number of objects to caption
        min_conf (float): confidence threshold
        img_size (int): YOLO inference size
        max_new_tokens (int): max caption tokens

    Returns:
        overlay (np.ndarray): RGB overlay image
        instances (list): list of dicts {label, conf, bbox, caption}
    """
    # convert PIL -> BGR numpy
    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h0, w0 = orig_bgr.shape[:2]

    # run segmentation (let YOLO handle resizing internally)
    results = seg_model(orig_bgr, imgsz=img_size)

    instances = []
    overlay = cv2.cvtColor(orig_bgr.copy(), cv2.COLOR_BGR2RGB)  # RGB for display

    for r in results:
        # masks
        masks = None
        try:
            masks = r.masks.data if getattr(r, "masks", None) is not None else None
        except Exception:
            masks = None

        boxes = r.boxes if getattr(r, "boxes", None) is not None else []

        for idx, box in enumerate(boxes):
            try:
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 1.0
                if conf < min_conf:
                    continue
                cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
                label = seg_model.names[cls_id] if cls_id >= 0 and seg_model.names else str(cls_id)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = [int(v) for v in xyxy]

                # mask if available
                mask_np = None
                if masks is not None:
                    try:
                        mask_np = masks[idx].cpu().numpy()
                    except Exception:
                        mask_np = None

                # crop
                pil_crop = None
                if mask_np is not None:
                    pil_crop = crop_instance_from_image(orig_bgr, mask_np, bbox=(x1, y1, x2, y2))
                else:
                    crop = orig_bgr[y1:y2+1, x1:x2+1]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_crop = Image.fromarray(crop_rgb)

                if pil_crop is None:
                    continue

                # caption
                inputs = processor(images=pil_crop, return_tensors="pt").to(device)
                with torch.no_grad():
                    if device.startswith("cuda"):
                        with torch.cuda.amp.autocast():
                            out = cap_model.generate(**inputs, max_new_tokens=max_new_tokens)
                    else:
                        out = cap_model.generate(**inputs, max_new_tokens=max_new_tokens)
                caption = processor.decode(out[0], skip_special_tokens=True)

                # overlay
                if mask_np is not None:
                    alpha = 0.4
                    overlay_mask = (mask_np > 0.5)
                    try:
                        overlay[overlay_mask] = (
                            overlay[overlay_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha
                        ).astype(np.uint8)
                    except Exception:
                        pass

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label} ({conf:.2f}): {caption}"
                cv2.putText(overlay, text, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                            1, cv2.LINE_AA)

                instances.append({
                    "label": label,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "caption": caption
                })

                if len(instances) >= max_instances:
                    break

            except Exception:
                continue

        if len(instances) >= max_instances:
            break

    # ensure same size as original
    if overlay.shape[:2] != (h0, w0):
        overlay = cv2.resize(overlay, (w0, h0))

    return overlay, instances
