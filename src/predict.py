"""
Predicción con el modelo entrenado.

Uso desde la terminal (raíz del repositorio):
    python src/predict.py ruta/a/mi_flor.jpg

También se importa desde la app de Streamlit (app/app.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "flower_cnn.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"
IMG_SIZE = (180, 180)  # debe ser el mismo tamaño usado al entrenar


def load_model_and_classes():
    """Carga el modelo entrenado y la lista de nombres de clases."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(img: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL en un lote de 1 imagen listo para el modelo.

    No se divide entre 255 porque la capa Rescaling del modelo ya lo hace.
    """
    arr = np.asarray(img.convert("RGB"), dtype="float32")
    arr = tf.image.resize(arr, IMG_SIZE).numpy()  # mismo método que en el entrenamiento
    return np.expand_dims(arr, axis=0)


def predict_image(img: Image.Image, model, class_names):
    """Devuelve (clase_predicha, confianza, vector_de_probabilidades)."""
    probs = model.predict(preprocess(img), verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python src/predict.py ruta/a/imagen.jpg")
        sys.exit(1)

    model, class_names = load_model_and_classes()
    label, confidence, _ = predict_image(Image.open(sys.argv[1]), model, class_names)
    print(f"Predicción: {label} ({confidence:.1%} de confianza)")
