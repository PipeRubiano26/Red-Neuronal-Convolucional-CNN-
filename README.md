#  Clasificador de flores con CNN

## Dataset

**Flowers Photos** (`tf_flowers`), publicado por el equipo de TensorFlow.

- ≈ 3.670 imágenes a color (JPG) de tamaño variable
- 5 clases: `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`
- Descarga automática desde `train.py` y desde el notebook (no se incluye en el repositorio por su tamaño)
- Partición estratificada: 70 % entrenamiento, 15 % validación, 15 % prueba

## Estructura del repositorio

```
proyecto-cnn/
├── notebooks/
│   └── 01_experimentacion.ipynb  
├── src/
│   ├── train.py                  
│   └── predict.py                 
├── app/
│   └── app.py                     
├── models/
│   ├── flower_cnn.keras           
│   └── class_names.json          
├── reports/figures/              
├── requirements.txt
└── README.md
```

## Instalación

Requiere **Python 3.10, 3.11 o 3.12**.

```bash
# 1. Crear y activar un entorno virtual
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac / Linux

# 2. Instalar dependencias
pip install -r requirements.txt
```
```bash
python src/train.py              # entrena (hasta 40 épocas) y guarda el modelo en models/
python src/train.py --epochs 20  # menos épocas
```
> Nota: ejecutar esto reentrena el modelo y reemplaza el de `models/`.

### Predecir una imagen desde la terminal
```bash
python src/predict.py ruta/a/mi_flor.jpg
```

### Aplicación Streamlit
```bash
streamlit run app/app.py
```
Se abre en el navegador (normalmente http://localhost:8501). Sube una imagen JPG o PNG y la app
muestra la **clase predicha**, su **nivel de confianza** y la probabilidad de cada clase.

## Arquitectura de la CNN

```
Input (180x180x3)
→ Data Augmentation (flip horizontal, rotación, zoom)   [solo en entrenamiento]
→ Rescaling (÷255)
→ Conv2D(32)  → MaxPooling
→ Conv2D(64)  → MaxPooling
→ Conv2D(128) → MaxPooling
→ Dropout(0.3) → Flatten
→ Dense(128, relu)
→ Dense(5, softmax)
```

- Pérdida: `sparse_categorical_crossentropy` · Optimizador: `adam`
- `EarlyStopping` sobre `val_loss` (paciencia 6, restaura los mejores pesos)

## Resultados

> Completa esta sección con los números de tu notebook.

| Métrica (conjunto de prueba) | Valor |
|---|---|
| Accuracy | 69% |
| F1 promedio (macro) | 66% |

| Modelo | Accuracy validación |
|---|---|
| Sin Data Augmentation | 97 % |
| Con Data Augmentation | 81 % |

**Curvas de entrenamiento**

![Curvas](reports/figures/curvas_entrenamiento.png)

**Matriz de confusión**

![Matriz de confusión](reports/figures/matriz_confusion.png)

**Predicciones de ejemplo**

![Predicciones](reports/figures/predicciones.png)

## Limitaciones

- El modelo solo conoce 5 tipos de flores: si se le da otra imagen (un perro, otra flor), igual responderá con una de las 5 clases.
- Es una CNN entrenada desde cero con pocas imágenes; usar *transfer learning* (por ejemplo MobileNetV2) probablemente mejoraría la accuracy.


