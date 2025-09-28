import os
import json
import random
from typing import List, Dict
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# -------- CONFIG --------
COCO_ROOT = r"C:\Users\24mdsml004\Desktop\Zidio\ImageCaptioningSegmentation\Dataset"
VAL_IMAGES = os.path.join(COCO_ROOT, "val2017")
ANN_VAL   = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")

# apna FT checkpoint path daalo (sirf best_epoch1 hai tumhare paas)
CKPT = os.path.join("..", "checkpoints", "blip_base_coco", "best_epoch1")

NUM_SAMPLES = 500     # fast eval. full = 5000 (val total). badha sakte ho.
MAX_NEW_TOKENS = 30   # generation length
SEED = 42

# -------- LOAD GT CAPTIONS --------
with open(ANN_VAL, "r", encoding="utf-8") as f:
    data = json.load(f)

id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
caps_per_img: Dict[int, List[str]] = {}
for ann in data["annotations"]:
    caps_per_img.setdefault(ann["image_id"], []).append(ann["caption"])

all_img_ids = list(caps_per_img.keys())
random.Random(SEED).shuffle(all_img_ids)
eval_ids = all_img_ids[:NUM_SAMPLES]

# -------- LOAD MODEL --------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(CKPT)
model = BlipForConditionalGeneration.from_pretrained(CKPT).to(device)
model.eval()

# -------- EVALUATE --------
hypotheses = []
references = []

def tokenize(s: str) -> List[str]:
    return s.lower().strip().split()

for i, img_id in enumerate(eval_ids, 1):
    file_name = id_to_file[img_id]
    img_path = os.path.join(VAL_IMAGES, file_name)

    refs = caps_per_img[img_id]
    references.append([tokenize(r) for r in refs])

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    hypo = processor.decode(out[0], skip_special_tokens=True)
    hypotheses.append(tokenize(hypo))

    if i % 50 == 0:
        print(f"[{i}/{NUM_SAMPLES}] sample done.")

# -------- BLEU (corpus) --------
smooth_fn = SmoothingFunction().method1
weights_1 = (1.0, 0, 0, 0)
weights_2 = (0.5, 0.5, 0, 0)
weights_3 = (0.33, 0.33, 0.33, 0)
weights_4 = (0.25, 0.25, 0.25, 0.25)

bleu1 = corpus_bleu(references, hypotheses, weights=weights_1, smoothing_function=smooth_fn)
bleu2 = corpus_bleu(references, hypotheses, weights=weights_2, smoothing_function=smooth_fn)
bleu3 = corpus_bleu(references, hypotheses, weights=weights_3, smoothing_function=smooth_fn)
bleu4 = corpus_bleu(references, hypotheses, weights=weights_4, smoothing_function=smooth_fn)

print("\n===== BLEU Scores (COCO val subset) =====")
print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")
print(f"BLEU-3: {bleu3:.4f}")
print(f"BLEU-4: {bleu4:.4f}")
