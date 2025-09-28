import os
import math
import torch
from torch.utils.data import DataLoader
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from data.coco_caption_dataset import COCODatasetCaptions
from tqdm import tqdm


# ======== CONFIG: paths & hyperparams ========
COCO_ROOT = r"C:\Users\24mdsml004\Desktop\Zidio\ImageCaptioningSegmentation\Dataset"
TRAIN_IMAGES = os.path.join(COCO_ROOT, "train2017")
VAL_IMAGES = os.path.join(COCO_ROOT, "val2017")
ANN_TRAIN = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")
ANN_VAL = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")

MODEL_NAME = "Salesforce/blip-image-captioning-base"
OUTPUT_DIR = os.path.join("..", "checkpoints", "blip_base_coco")

# memory-safe for 6GB:
EPOCHS = 3
TRAIN_BATCH = 1
GRAD_ACCUM = 16
VAL_BATCH = 2
LR = 5e-5
WARMUP_RATIO = 0.03
NUM_WORKERS = 0  # Windows-safe
FP16 = True
MAX_TOKENS = 40

# logging config
PRINT_EVERY_UPDATES = 10  # print every N optimizer updates


def main():
    # ======== Accelerator (GPU preferred) ========
    accelerator = Accelerator(
        mixed_precision="fp16" if (FP16 and torch.cuda.is_available()) else "no",
        device_placement=True
    )
    accelerator.print(f"⚡ Using device: {accelerator.device}")

    # ======== Load model/processor ========
    processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        use_safetensors=True,
    )

    # Freeze vision encoder, train text decoder only
    for name, p in model.named_parameters():
        if not (name.startswith("text_decoder") or "lm_head" in name):
            p.requires_grad = False

    # ======== Datasets ========
    train_ds_py = COCODatasetCaptions(TRAIN_IMAGES, ANN_TRAIN, use_all_captions=False)
    val_ds_py = COCODatasetCaptions(VAL_IMAGES, ANN_VAL, use_all_captions=False)

    def collate_fn(batch):
        images = [b["image"] for b in batch]
        texts = [b["caption"] for b in batch]
        enc = processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_TOKENS,
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc

    train_loader = DataLoader(
        train_ds_py, batch_size=TRAIN_BATCH, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds_py, batch_size=VAL_BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # total steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    num_warmup_steps = int(total_training_steps * WARMUP_RATIO)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps
    )

    # prepare everything for accelerator (model + optimizer + dataloaders + scheduler)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ======== Training loop ========
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss_raw = 0.0
        running_updates = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for step, batch in progress_bar:
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs.loss
                loss_for_backward = loss / GRAD_ACCUM

            accelerator.backward(loss_for_backward)

            running_loss_raw += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                running_updates += 1

                if global_step % PRINT_EVERY_UPDATES == 0:
                    avg_loss_per_raw_step = running_loss_raw / (GRAD_ACCUM * running_updates)
                    accelerator.print(
                        f"Epoch {epoch+1} | Global Step {global_step} | "
                        f"Updates {running_updates} | avg_loss={avg_loss_per_raw_step:.4f}"
                    )

                progress_bar.set_postfix({
                    "global_step": global_step,
                    "updates": running_updates,
                    "last_loss": f"{loss.item():.4f}"
                })

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation epoch {epoch+1}"):
                with accelerator.autocast():
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    val_batches += 1
        val_loss = val_loss / max(1, val_batches)
        accelerator.print(f"Epoch {epoch+1} done. Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            unwrapped = accelerator.unwrap_model(model)
            save_path = os.path.join(OUTPUT_DIR, f"best_epoch{epoch+1}")
            unwrapped.save_pretrained(save_path, safe_serialization=True)
            processor.save_pretrained(save_path)
            accelerator.print(f"✅ Saved best to {save_path}")

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(os.path.join(OUTPUT_DIR, "last"), safe_serialization=True)
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "last"))
    accelerator.print("🎉 Training complete.")


if __name__ == "__main__":
    main()
