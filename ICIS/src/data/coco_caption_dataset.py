import os
import json
from typing import List, Dict, Any
from PIL import Image
from torch.utils.data import Dataset

class COCODatasetCaptions(Dataset):
    """
    COCO 2017 captions dataset.
    images_root: r'D:\coco\train2017'  (or val2017)
    ann_file:    r'D:\coco\annotations\captions_train2017.json'
    """
    def __init__(self, images_root: str, ann_file: str, use_all_captions: bool = False):
        self.images_root = images_root
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Map img_id -> filename
        self.imgs = {img["id"]: img["file_name"] for img in data["images"]}

        # Collect captions per img_id
        caps_per_img = {}
        for ann in data["annotations"]:
            caps_per_img.setdefault(ann["image_id"], []).append(ann["caption"])

        self.samples = []
        for img_id, caps in caps_per_img.items():
            file_name = self.imgs.get(img_id)
            if not file_name:
                continue
            img_path = os.path.join(images_root, file_name)
            if not os.path.exists(img_path):
                # skip missing files
                continue
            if use_all_captions:
                for c in caps:
                    self.samples.append({"image_path": img_path, "caption": c})
            else:
                # pick first caption per image (memory friendly)
                self.samples.append({"image_path": img_path, "caption": caps[0]})

        print(f"[COCO] Loaded {len(self.samples)} samples from {images_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        caption = sample["caption"]
        return {"image": image, "caption": caption}
