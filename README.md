# AI Crop Disease Detection & Advisory System

A student project using Python, Flask, TensorFlow/Keras, MobileNetV2, HTML/CSS, and SQLite to screen crop-leaf images for classes learned by a trained model.

## Features

- Upload JPG, JPEG, PNG, or WEBP leaf images (maximum 8 MB).
- Basic image validation and brightness/size warnings.
- Top prediction, confidence score, and top-3 alternatives.
- Low-confidence warning.
- Educational prevention and treatment guidance.
- SQLite prediction history and dashboard.

> **Important:** This is an educational screening tool, not a professional diagnosis. Model confidence is not a guarantee of correctness. The built-in image-quality checks do not verify that an image contains a leaf. Consult local agricultural experts before applying treatments or pesticides.

## 1. Requirements

- Python 3.10–3.12 recommended; use a TensorFlow-supported Python version for your operating system.
- Internet connection for installing dependencies and downloading pretrained MobileNetV2 weights.
- A plant disease dataset with one class per folder. For example, a PlantVillage-derived dataset.

## 2. Create a virtual environment (Windows PowerShell)

```powershell
cd ai_crop_disease_system
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Prepare dataset

Create this structure (class folder names become prediction labels):

```text
dataset/
└── PlantVillage/
    ├── Tomato___healthy/
    ├── Tomato___Early_blight/
    ├── Tomato___Late_blight/
    └── ...
```

Place image files inside each class folder. Keep the dataset out of GitHub. Ensure you have permission to use the dataset and check its license/terms.

## 4. Train the model

```powershell
python model/train.py
```

The training script creates:
- `model/crop_disease_model.keras`
- `model/class_names.json`

Training may take time and may be slow on CPU. The script uses transfer learning with a frozen MobileNetV2 base for a starter implementation.

## 5. Run the web application

```powershell
python app.py
```

Open the local address shown in the terminal, usually `http://127.0.0.1:5000`.

The trained model files must exist before image predictions can run. Without them, the application displays an instruction to train the model first.

## Project structure

```text
ai_crop_disease_system/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   └── train.py
├── dataset/                 # create locally; ignored by Git
├── database/                # SQLite database is created automatically
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── dashboard.html
│   └── history_detail.html
└── static/
    ├── css/style.css
    ├── js/main.js
    └── uploads/
```

## Evaluation suggestions

For your project report, evaluate a separate held-out test set. Report accuracy, per-class precision/recall/F1-score, a confusion matrix, and results on real-world photos. Avoid claiming a performance number before running your own evaluation.

## Possible future improvements

- Separate train/validation/test split and class-imbalance handling.
- Grad-CAM visual explanations.
- Better blur and leaf-region detection.
- Authentication and user-specific history.
- Verified, region-specific advisory content from agricultural extension sources.
- Deployment with environment variables, upload limits, HTTPS, and debug mode disabled.
