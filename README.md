
---

# 🖼️ Zidio: Image Captioning & Image Segmentation

**Zidio** is an AI-powered project combining **image captioning** and **image segmentation**. It allows users to generate descriptive captions for images and detect object regions using deep learning models. This project is lightweight, easy to set up, and designed for experimentation and local deployment.

Large files like pre-trained checkpoints and datasets are **not tracked in GitHub** and should be downloaded separately via the shared drive link.

---

## Project Structure

* **ICIS/src/** – Core source code
* **ICIS/src/app/** – Web/GUI applications
* **ICIS/src/data/** – Dataset handling scripts
* **ICIS/src/runs/** – Prediction outputs (ignored in Git)
* **playground.py** – Experimental scripts
* **requirements.txt** – Python dependencies
* **README.md** – Project documentation
* **yolov8n-seg.pt** – Small model file tracked with Git LFS

> Large folders such as `ICIS/checkpoints/`, `runs/`, and `Datasets/` are ignored in Git to prevent exceeding GitHub file size limits.

---

## Features

* **Image Captioning** – Automatically generate captions for images
* **Image Segmentation** – Detect and segment objects using YOLOv8 models
* **Web Interface** – Interactive apps for local testing
* **Modular Scripts** – Flexible scripts to train, evaluate, and infer results

---

## Setup Instructions

1. Clone the repository:

```
git clone https://github.com/A1610/Zidio_Image-Captioning_Image-Segmentation.git
cd Zidio_Image-Captioning_Image-Segmentation
```

2. Create and activate a virtual environment:

```
python -m venv mlenv
.\mlenv\Scripts\activate  # Windows
source mlenv/bin/activate # Linux/Mac
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Download large files (checkpoints & datasets):

[Download Large Files]( https://drive.google.com/drive/folders/1KnVl2-n5gpDN9Kyrohaq_za5nzBKZ23E?usp=sharing )

* Place `ICIS/checkpoints/` and `Datasets/` in the project root.

---

## Usage

1. **Run the Web App**:

```
python ICIS/src/app/app.py
python ICIS/src/app/app_webcam.py
```

2. **Train Captioning Model**:

```
python ICIS/src/train_caption_blip.py
```

3. **Run Segmentation Inference**:

```
python ICIS/src/segmentation_baseline.py
```

4. **Generate Captions**:

```
python ICIS/src/infer_caption_ckpt.py
```

---

## Notes

* Large files (>100 MB) are **not tracked** in GitHub; use the drive link.
* `runs/` folder stores predictions and is ignored.
* `mlenv/` virtual environment is ignored.
* Small models like `yolov8n-seg.pt` are tracked with Git LFS.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---


