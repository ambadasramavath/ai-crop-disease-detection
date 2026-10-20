import os, json, sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageStat
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "model" / "crop_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
DB_PATH = BASE_DIR / "database" / "predictions.db"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
CONFIDENCE_THRESHOLD = 0.60

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-for-deployment")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

ADVISORIES = {
    "healthy": {
        "summary": "No disease was identified among the classes supported by this model.",
        "prevention": "Continue regular scouting, maintain good airflow, water at the soil level, and clean tools between plants.",
        "treatment": "No disease treatment is indicated by this prediction. Continue monitoring the plant."
    },
    "default": {
        "summary": "This is an AI-generated screening result, not a confirmed diagnosis.",
        "prevention": "Inspect nearby plants, remove plant debris where appropriate, avoid overhead watering when possible, and keep tools clean.",
        "treatment": "Compare symptoms with a trusted local agricultural extension resource before taking action. Do not apply pesticides based on this prediction alone."
    }
}

_model = None
_class_names = None

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                warning TEXT
            )
        """)
init_db()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def check_image_quality(image):
    """Basic checks only; these do not prove that an image contains a crop leaf."""
    if image.width < 80 or image.height < 80:
        return "Image is very small. Upload a clearer image at least 80 × 80 pixels."
    rgb = image.convert("RGB")
    stats = ImageStat.Stat(rgb)
    brightness = sum(stats.mean) / 3
    if brightness < 18:
        return "Image appears very dark. Try taking the photo in better lighting."
    if brightness > 245:
        return "Image appears overexposed. Try softer, indirect daylight."
    return None

def load_model():
    global _model, _class_names
    if _model is None:
        if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
            raise RuntimeError(
                "Trained model files are missing. Follow README.md to train the model "
                "and create model/crop_disease_model.keras and model/class_names.json."
            )
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        _class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return _model, _class_names

def predict_image(image_path):
    model, class_names = load_model()
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    # MobileNetV2 preprocessing maps pixels from [0, 255] to [-1, 1].
    array = array / 127.5 - 1.0
    probabilities = model.predict(array, verbose=0)[0]
    best_indices = np.argsort(probabilities)[::-1][:3]
    results = [
        {"label": class_names[int(i)], "confidence": float(probabilities[i])}
        for i in best_indices
    ]
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        upload = request.files.get("leaf_image")
        if not upload or not upload.filename:
            flash("Please choose a leaf image first.", "error")
            return redirect(url_for("index"))
        if not allowed_file(upload.filename):
            flash("Unsupported file type. Upload PNG, JPG, JPEG, or WEBP.", "error")
            return redirect(url_for("index"))

        safe_name = secure_filename(upload.filename)
        unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{safe_name}"
        destination = UPLOAD_DIR / unique_name
        upload.save(destination)

        try:
            with Image.open(destination) as img:
                img.verify()
            with Image.open(destination) as img:
                quality_warning = check_image_quality(img)
        except Exception:
            destination.unlink(missing_ok=True)
            flash("The uploaded file is not a valid image.", "error")
            return redirect(url_for("index"))

        if quality_warning:
            destination.unlink(missing_ok=True)
            flash(quality_warning, "error")
            return redirect(url_for("index"))

        try:
            predictions = predict_image(destination)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            flash(str(exc), "error")
            return redirect(url_for("index"))

        top = predictions[0]
        warning = None
        if top["confidence"] < CONFIDENCE_THRESHOLD:
            warning = (
                "Low-confidence result. The model is uncertain; retake the image or "
                "seek advice from a local agricultural expert."
            )

        label_lower = top["label"].lower()
        advisory = ADVISORIES["healthy"] if "healthy" in label_lower else ADVISORIES["default"]
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO predictions (filename, predicted_class, confidence, created_at, warning) VALUES (?, ?, ?, ?, ?)",
                (unique_name, top["label"], top["confidence"],
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), warning)
            )
            record_id = cursor.lastrowid

        return render_template(
            "result.html", prediction=top, alternatives=predictions[1:],
            warning=warning, advisory=advisory, image_url=url_for("static", filename=f"uploads/{unique_name}"),
            record_id=record_id
        )
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        uncertain = conn.execute("SELECT COUNT(*) FROM predictions WHERE warning IS NOT NULL AND warning != ''").fetchone()[0]
    return render_template("dashboard.html", rows=rows, total=total, uncertain=uncertain)

@app.route("/history/<int:record_id>")
def history_detail(record_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        flash("Prediction record not found.", "error")
        return redirect(url_for("dashboard"))
    return render_template("history_detail.html", row=row)

@app.errorhandler(413)
def too_large(_error):
    flash("Image is too large. Maximum upload size is 8 MB.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
