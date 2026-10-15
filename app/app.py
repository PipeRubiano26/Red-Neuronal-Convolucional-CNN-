"""
Aplicación Streamlit: sube una foto de una flor y el modelo predice su clase.

Ejecutar desde la raíz del repositorio:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.predict import MODEL_PATH, load_model_and_classes, predict_image  # noqa: E402

st.set_page_config(page_title="Clasificador de flores")
st.title("Clasificador de flores con CNN")
st.write(
    "Sube una foto y el modelo dirá de qué tipo de flor se trata. "
    "Solo conoce 5 clases: **daisy, dandelion, roses, sunflowers y tulips**."
)

if not MODEL_PATH.exists():
    st.error(
        f"No encuentro el modelo en `{MODEL_PATH}`. "
        "Entrena el modelo (notebook o `python src/train.py`) y copia `flower_cnn.keras` "
        "y `class_names.json` a la carpeta `models/`."
    )
    st.stop()


@st.cache_resource  
def get_model():
    return load_model_and_classes()


model, class_names = get_model()

file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Analizando..."):
        label, confidence, probs = predict_image(img, model, class_names)

    st.success(f"Predicción: **{label}**  —  confianza: **{confidence:.1%}**")
    if confidence < 0.5:
        st.warning(
            "La confianza es baja: el modelo no está seguro. "
            "Puede que la imagen no sea de una de las 5 clases que conoce."
        )

    st.subheader("Probabilidad por clase")
    df = pd.DataFrame({"Clase": class_names, "Probabilidad": probs}).set_index("Clase")
    st.bar_chart(df)
