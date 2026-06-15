# Dart Detector App (YOLO26s)

This is a self-contained, portable, and easy-to-use repository for real-time dart detection using a state-of-the-art **YOLO26** model. It includes a pre-trained model, a live webcam testing script with a symmetrical 1:1 aspect ratio HUD, and a template directory structure to train a custom model on your own dataset.

---

## 📁 Repository Structure

```text
dart-detector-package/
├── models/
│   └── pretrained/
│       └── best.pt           # Pre-trained YOLO26s model weights (default)
├── trained_models/
│   └── .gitkeep              # Placeholder folder for your custom trained models
├── dataset/
│   ├── images/
│   │   ├── train/            # Put your custom training images here
│   │   └── val/              # Put your custom validation images here
│   ├── labels/
│   │   ├── train/            # Put your custom training YOLO label text files here
│   │   └── val/              # Put your custom validation YOLO label text files here
│   └── dataset.yaml          # Dataset path configuration file
├── requirements.txt          # Python dependencies
├── train.py                  # Script to train a new YOLO26 model
├── detect_live.py            # Script for live real-time webcam detection
└── README.md                 # This instruction guide
```

---

## ⚙️ Installation & Setup

Ensure you have **Python 3.9 - 3.12** installed on your system.

### 1. Create a Virtual Environment (Recommended)
Open your terminal inside this folder and run:
```cmd
python -m venv env
env\Scripts\activate
```

### 2. Install PyTorch with GPU Support (CUDA)
To speed up training and inference using your NVIDIA GPU (e.g., RTX 3060/4060), install the CUDA-enabled version of PyTorch:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Package Dependencies
Install the remaining packages (`ultralytics` and `opencv-python`):
```cmd
pip install -r requirements.txt
```

---

## 🎯 1. How to Test the Existing Model (Live Webcam)

To run the live detection script using the pre-trained model in `models/pretrained/best.pt`, execute:
```cmd
python detect_live.py
```

### 🎮 Live Feed Controls:
* **`q`** or **`Q`** or **`ESC`**: Gracefully close the window and exit the application.
* **`w`** or **`+`**: **Increase** the confidence threshold by 5% (makes it stricter, reducing false positives).
* **`s`** or **`-`**: **Decrease** the confidence threshold by 5% (makes it more sensitive).

### 🖥️ Interactive HUD Display:
* **Red Target Dot:** The exact center coordinate `(X, Y)` of the dart, printed next to the dot.
* **Relative Coordinate `[rel_X, rel_Y]`:** Symmetrical normalized values from `0.00` to `1.00`. If you know the physical dimensions of the frame (e.g., 0.5m x 0.5m), multiply these values to get the exact physical coordinates (e.g., `0.5 * 0.5m = 0.25m`).
* **Centered 1:1 Square Crop:** The script dynamically center-crops the feed into a square shape, maximizing your hardware field of view (FOV) while keeping spatial scaling equal on both axes.

---

## 🏋️ 2. How to Train a New Model on Your Data

If you want to train a custom model to detect darts (or other items) in your specific environment:

### Step A: Prepare Your Dataset
1. Place your training images in `dataset/images/train/` and validation images in `dataset/images/val/`.
2. Convert your bounding box annotations to **YOLO format** (normalized coordinates `class_id x_center y_center width height`).
3. Place matching `.txt` annotation files in `dataset/labels/train/` and `dataset/labels/val/` respectively (e.g., `image_01.jpg` must match `image_01.txt`).

### Step B: Start the Training
Run the training script:
```cmd
python train.py
```
* The script automatically checks if your GPU (CUDA) is available and trains the model for **100 epochs** at a **batch size of 16** and image size of **640**.
* If you do not have a GPU, it will fallback to CPU (much slower).

---

## 🔄 3. Where are New Models Saved & How to Update Detection

When you train a new model:
1. The training run outputs will be saved inside the **`trained_models/`** directory in a dedicated folder timestamped with the run start time (e.g., `trained_models/dart_yolo26s_20260615_191500/`).
2. The newly trained best weights will be saved inside that run folder at:
   ```text
   trained_models/dart_yolo26s_YYYYMMDD_HHMMSS/weights/best.pt
   ```
3. **To update the live detection script with your new model, either:**
   * **Option A:** Copy your new `best.pt` file from `trained_models/dart_yolo26s_YYYYMMDD_HHMMSS/weights/` and overwrite the file in `models/pretrained/best.pt`. (Recommended)
   * **Option B:** Open `detect_live.py` and modify the `MODEL_PATH` variable to point to your new run folder path:
     ```python
     MODEL_PATH = os.path.join(script_dir, "trained_models", "dart_yolo26s_YYYYMMDD_HHMMSS", "weights", "best.pt")
     ```

---

## 🛠️ Troubleshooting

### Multiple Webcams / Wrong Camera Active
If the script opens the wrong camera (e.g., your laptop's built-in webcam instead of your external USB camera), you can change the camera source index:
1. Open [detect_live.py](file:///D:/Projects/dart-finder-bot/dart-detector-package/detect_live.py).
2. Locate the line:
   ```python
   cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
   ```
3. Change the integer index `0` to `1` (or `2`, depending on how many cameras are connected):
   ```python
   cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
   ```

