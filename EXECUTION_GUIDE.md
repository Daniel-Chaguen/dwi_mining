# 📖 Guía de Ejecución del Proyecto

Esta guía detalla paso a paso cómo ejecutar el proyecto de clasificación de trastornos neuropsiquiátricos usando imágenes DWI.

---

## 🔧 Configuración del Entorno

### Paso 1: Instalar Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar paquetes base
pip install numpy pandas matplotlib seaborn jupyter
pip install scikit-learn xgboost

# Instalar herramientas de neuroimagen
pip install nibabel dipy

# Instalar PyTorch (ajustar según tu hardware)
# CPU:
pip install torch torchvision

# GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Paso 2: Verificar Instalación

```python
import torch
import nibabel as nib
import dipy
import sklearn

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Nibabel version: {nib.__version__}")
print(f"DIPY version: {dipy.__version__}")
```

---

## 📥 Descarga de Datos

### Opción 1: Google Colab (Recomendada)

1. Abre `scripts/data_download.ipynb` en Google Colab
2. Monta tu Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Actualiza la variable `BASE_PATH` con tu ruta de Google Drive
4. Ejecuta todas las celdas para descargar el dataset completo

### Opción 2: Descarga Manual

1. Visita [OpenNeuro](https://openneuro.org/datasets/ds000030/versions/1.0.0)
2. Descarga el dataset completo (⚠️ ~70 GB)
3. Organiza los archivos según estructura BIDS:
   ```
   data/
   ├── sub-10159/
   │   └── dwi/
   │       ├── sub-10159_dwi.nii.gz
   │       ├── sub-10159_dwi.bval
   │       └── sub-10159_dwi.bvec
   ├── sub-10171/
   ...
   ```

---

## 🚀 Ejecución del Pipeline Completo

### Fase 1 y 2: Entendimiento de Datos

```bash
# Ejecutar notebook de análisis exploratorio
jupyter notebook mineria_fase2.ipynb
```

**Salidas esperadas:**
- Distribución de diagnósticos
- Estadísticas de metadatos DWI
- Visualización 3D de volúmenes de ejemplo

---

### Fase 3: Preparación de Datos

#### A. Generar Características FA (Bloques)

```bash
jupyter notebook mineria_fase3.ipynb
```

**Proceso:**
1. Carga de volúmenes DWI (96×96×50×64)
2. Cálculo del modelo de tensor de difusión (DTI)
3. Extracción de Anisotropía Fraccional (FA)
4. División en bloques 8×8×8 (512 features)
5. Guardado en `out/dwi_block_features_for_svm.csv`

**⏱️ Tiempo estimado:** 2-3 horas (200 sujetos)

#### B. Preprocesar Volúmenes para CNN

El mismo notebook `mineria_fase3.ipynb` también:
1. Filtra volúmenes por dimensiones (96×96×50)
2. Normaliza con z-score por sujeto
3. Guarda bloques pickle en `out/data/block_*.pkl`
4. Genera `manifest.csv` con índices

---

### Fase 4: Modelado

#### Modelo 1: SVM con Características FA

```python
# En mineria_fase4.ipynb (sección "Modelos Clásicos")
# Ejecutar celdas de entrenamiento SVM:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Pipeline: Scaling → SVM
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced"))
])

# Grid Search
param_grid = {
    "svm__kernel": ["rbf", "linear"],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto", 0.01, 0.1]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="balanced_accuracy")
grid_search.fit(X_train, y_train)

# Guardar mejor modelo
best_model = grid_search.best_estimator_
```

**Salidas:**
- `out/svm_model.pkl`
- Mejores hiperparámetros
- Balanced Accuracy en validación cruzada



#### Modelo 2: CNN 3D

```python
# En mineria_fase4.ipynb (sección "CNN 3D")
# 1. Cargar volúmenes preprocesados
volumes, labels = load_all_blocks_to_ram()

# 2. Instanciar modelo
model = ImprovedCNN3D(in_channels=64, num_classes=2)

# 3. Entrenar
train(
    volumes=volumes,
    labels=labels,
    out_dir=OUT_FOLDER,
    instance_model=model,
    model_name="v2_cnn"
)
```

**Configuración de Entrenamiento:**
- Batch size: 8
- Epochs: 150 (con early stopping por weighted accuracy)
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss
- Regularización: Dropout (0.1-0.2), BatchNorm

**⏱️ Tiempo de entrenamiento:**
- CPU: ~10-15 horas
- GPU (RTX 3070): ~2-3 horas

**Salidas:**
- `out/best_v2_cnn.pth` (mejor checkpoint)
- `out/v2_cnn.pth` (última época)
- `out/v2_cnn.csv` (histórico de entrenamiento)
- `out/v2_cnn_training.png` (curvas de loss/accuracy)

---

### Fase 5: Evaluación

```bash
jupyter notebook mineria_fase5.ipynb
```

**Análisis incluidos:**

1. **Métricas Cuantitativas**
   - Confusion matrix
   - Classification report (precision, recall, F1)
   - Balanced accuracy

2. **Visualizaciones t-SNE**
   - Espacio latente del SVM (features FA)
   - Embeddings de la CNN (features de block4)
   - Comparación etiquetas reales vs predicciones

3. **Análisis de Activaciones (CNN)**
   - Mapas de activación por capa
   - Comparación input original vs features aprendidos
   - Visualización de transformación progresiva

4. **Análisis de Errores**
   - Ejemplos correctamente clasificados
   - Ejemplos mal clasificados
   - Distribución de confianza (probabilidades)

**Salidas:**
- Gráficas de matrices de confusión
- t-SNE plots (PNG)
- Mapas de activación (PNG)
- Reporte de métricas (texto)

---

