import json
import random
import cv2
import os
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

# ===== CONFIG =====
# apna dataset ka root path yaha set karo:
COCO_ROOT = r"D:\coco"

IMG_DIR = os.path.join(COCO_ROOT, "val2017")  # validation images
ANN_FILE_CAP = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")
ANN_FILE_SEG = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

# ===== CAPTION LOAD =====
coco_caps = COCO(ANN_FILE_CAP)
img_ids = coco_caps.getImgIds()
img_meta = coco_caps.loadImgs(random.choice(img_ids))[0]

# Image path
img_path = os.path.join(IMG_DIR, img_meta['file_name'])
print("🖼️ Image:", img_path)

# Load captions
ann_ids = coco_caps.getAnnIds(imgIds=img_meta['id'])
anns = coco_caps.loadAnns(ann_ids)
captions = [a['caption'] for a in anns]
print("📝 Captions:", captions)

# ===== SEGMENTATION LOAD =====
coco_seg = COCO(ANN_FILE_SEG)
seg_ann_ids = coco_seg.getAnnIds(imgIds=img_meta['id'])
seg_anns = coco_seg.loadAnns(seg_ann_ids)

# ===== IMAGE LOAD (safe way) =====
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

image_data = cv2.imread(img_path)
if image_data is None:
    raise ValueError(f"❌ Failed to load image with OpenCV: {img_path}")

img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

# ===== PLOT WITH SEGMENTATION =====
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')

# Draw segmentation masks (just overlay contours)
for ann in seg_anns:
    mask = coco_seg.annToMask(ann)
    img[mask == 1] = (0, 255, 0)  # green overlay

plt.imshow(img)
plt.title("Random COCO Image with Captions")
plt.show()
