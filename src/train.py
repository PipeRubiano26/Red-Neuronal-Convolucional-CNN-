"""
Entrena la CNN de clasificación de flores y guarda el modelo.

Uso (desde la raíz del repositorio):
    python src/train.py
    python src/train.py --epochs 40

Qué hace, en orden:
    1. Descarga el dataset (flower_photos) si no está descargado.
    2. Divide los datos en entrenamiento / validación / prueba (70/15/15).
    3. Construye la CNN (con Data Augmentation) y la entrena.
    4. Evalúa en el conjunto de prueba (métricas + matriz de confusión).
    5. Guarda el modelo en models/ y las gráficas en reports/figures/.
"""
import argparse
import json
import pathlib
import random

import matplotlib

matplotlib.use("Agg")  # guarda gráficas en archivos, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 42


def download_dataset():
    """Descarga el dataset y devuelve (rutas de imágenes, etiquetas numéricas, nombres de clases)."""
    archive = tf.keras.utils.get_file(origin=URL, extract=True)
    data_dir = pathlib.Path(archive).with_suffix("")
    if not data_dir.exists():
        data_dir = next(pathlib.Path(archive).parent.rglob("flower_photos"))

    paths = sorted(data_dir.glob("*/*.jpg"))
    labels = [p.parent.name for p in paths]
    class_names = sorted(set(labels))
    y = np.array([class_names.index(label) for label in labels])
    print(f"{len(paths)} imágenes | clases: {class_names}")
    return paths, y, class_names


def split_data(paths, y):
    """70% entrenamiento, 15% validación, 15% prueba (estratificado por clase)."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    print(f"train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)  # todas las imágenes al mismo tamaño
    return img, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def make_augmentation():
    """Variaciones aleatorias de las imágenes. Solo se activan al entrenar."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def build_model(num_classes, use_augmentation=True):
    """CNN: 3 bloques Conv+Pool, Dropout, y capas densas con softmax al final."""
    model_layers = [keras.Input(shape=(*IMG_SIZE, 3))]
    if use_augmentation:
        model_layers.append(make_augmentation())
    model_layers += [
        # La normalización va DENTRO del modelo: así la app no tiene que acordarse de hacerla.
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
    model = keras.Sequential(model_layers)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["accuracy"], label="entrenamiento")
    ax[0].plot(history.history["val_accuracy"], label="validación")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Época")
    ax[0].legend()
    ax[1].plot(history.history["loss"], label="entrenamiento")
    ax[1].plot(history.history["val_loss"], label="validación")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Época")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "curvas_entrenamiento.png", dpi=150)
    plt.close(fig)


def evaluate(model, test_ds, y_test, class_names):
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest -> loss: {loss:.4f} | accuracy: {acc:.2%}\n")

    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False
    )
    ax.set_title("Matriz de confusión (conjunto de prueba)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matriz_confusion.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Entrena la CNN de flores.")
    parser.add_argument("--epochs", type=int, default=40, help="máximo de épocas")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths, y, class_names = download_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(paths, y)

    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

    model = build_model(len(class_names), use_augmentation=True)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early_stop]
    )

    plot_history(history)
    evaluate(model, test_ds, y_test, class_names)

    model.save(MODELS_DIR / "flower_cnn.keras")
    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f)
    print(f"\nModelo guardado en {MODELS_DIR / 'flower_cnn.keras'}")


if __name__ == "__main__":
    main()
