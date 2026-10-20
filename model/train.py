"""
Train MobileNetV2 with a folder-based dataset.

Expected dataset structure:
dataset/PlantVillage/
  Tomato___healthy/
    image1.jpg
  Tomato___Early_blight/
    image2.jpg
  ...

Use a dataset you are permitted to use. Keep the dataset out of Git.
"""
import json
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "PlantVillage"
MODEL_DIR = ROOT / "model"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def main():
    if not DATA_DIR.exists():
        raise SystemExit(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Download/extract a plant disease dataset and place class folders inside dataset/PlantVillage/."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    # Match preprocessing used by app.py: scale 0..255 to -1..1.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "crop_disease_model.keras"),
            save_best_only=True, monitor="val_loss"
        )
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # Evaluate on validation split; for a formal report, keep a separate test set.
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Saved model to {MODEL_DIR / 'crop_disease_model.keras'}")
    print("Note: validation accuracy is not a substitute for testing on real farm images.")

if __name__ == "__main__":
    main()
