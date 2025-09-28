from ultralytics import YOLO
import cv2

# 1) Pretrained segmentation model load
model = YOLO("yolov8n-seg.pt")  # nano seg model (lightweight, fast)

# 2) Run on sample image (same image used for captioning)
img_path = r"C:\Users\sriha\OneDrive\Desktop\images.jpg"
results = model(img_path, save=True)  # save=True => output file will be saved

# 3) Print detected objects
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Object: {model.names[cls_id]} | Confidence: {conf:.2f}")

print("✅ Segmentation complete. Check 'runs/segment/predict/' folder for output image.")
