# 🔬 Technical Details - DWI Mining Project

Este documento proporciona detalles técnicos profundos sobre las metodologías, arquitecturas y decisiones de diseño del proyecto.

---

## 📐 Feature Engineering: FA Block Extraction

### Fractional Anisotropy (FA) Mathematics

La **Anisotropía Fraccional** es una métrica derivada del tensor de difusión que cuantifica la direccionalidad de la difusión del agua en tejido cerebral.

#### Paso 1: Estimación del Tensor de Difusión

El tensor de difusión **D** es una matriz simétrica 3×3 que describe cómo las moléculas de agua se difunden en el espacio:

$$
D = \begin{bmatrix}
D_{xx} & D_{xy} & D_{xz} \\
D_{xy} & D_{yy} & D_{yz} \\
D_{xz} & D_{yz} & D_{zz}
\end{bmatrix}
$$

Se obtiene ajustando el modelo DTI a los datos DWI mediante:

$$
S(\mathbf{b}, \mathbf{g}) = S_0 \exp\left(-b \, \mathbf{g}^T D \mathbf{g}\right)
$$

Donde:
- $S(\mathbf{b}, \mathbf{g})$ = intensidad medida con gradiente $\mathbf{g}$ y valor b
- $S_0$ = imagen sin gradiente de difusión (b0)
- $b$ = factor de ponderación de difusión (b-value)
- $\mathbf{g}$ = vector unitario de dirección del gradiente

#### Paso 2: Descomposición Espectral

El tensor se descompone en autovalores y autovectores:

$$
D = V \Lambda V^{-1}
$$

Donde:
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \lambda_3)$ con $\lambda_1 \geq \lambda_2 \geq \lambda_3$
- $\lambda_1$ = difusión máxima (dirección principal de fibras)
- $\lambda_2, \lambda_3$ = difusión perpendicular

#### Paso 3: Cálculo de FA

$$
FA = \sqrt{\frac{3}{2}} \frac{\sqrt{(\lambda_1 - \bar{\lambda})^2 + (\lambda_2 - \bar{\lambda})^2 + (\lambda_3 - \bar{\lambda})^2}}{\sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}
$$

Donde $\bar{\lambda} = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}$ es la difusividad media.

**Interpretación:**
- **FA = 0**: Difusión isotrópica (líquido cefalorraquídeo, sustancia gris)
- **FA = 1**: Difusión perfectamente direccional (fibras de sustancia blanca bien organizadas)
- **Rango típico en cerebro**: 0.2 - 0.8

### Block Aggregation Strategy

En lugar de usar todos los 4,608,000 voxels (96 × 96 × 50) como features, implementamos agregación por bloques:

1. **División espacial**: Crear grid 3D de 8×8×8 = **512 bloques**
2. **Agregación local**: Calcular $\bar{FA}_{\text{block}} = \frac{1}{N} \sum_{v \in \text{block}} FA(v)$
3. **Resultado**: Vector de 512 features por sujeto

**Ventajas:**
- ✅ Reduce dimensionalidad 9000× (4.6M → 512)
- ✅ Robustez ante ruido voxel-wise
- ✅ Preserva información espacial regional
- ✅ Compatible con modelos clásicos (SVM, XGBoost)

---

## 🧠 3D CNN Architecture Deep Dive

### ImprovedCNN3D Architecture

```
Input: (Batch, 64, 50, 96, 96)
       [B, Channels, Depth, Height, Width]

┌─────────────────────────────────────────────────┐
│ Block 1: Feature Extraction                     │
│   Conv3D(64→32, k=3, s=1, p=1)                  │
│   BatchNorm3D(32)                               │
│   ReLU(inplace=True)                            │
│   Dropout3D(p=0.1)                              │
│ Output: (B, 32, 50, 96, 96)                     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Block 2: Spatial Reduction                      │
│   Conv3D(32→64, k=3, s=1, p=1)                  │
│   BatchNorm3D(64)                               │
│   ReLU(inplace=True)                            │
│   MaxPool3D(k=2, s=2)                           │
│   Dropout3D(p=0.1)                              │
│ Output: (B, 64, 25, 48, 48)                     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Block 3: High-Level Features                    │
│   Conv3D(64→128, k=3, s=1, p=1)                 │
│   BatchNorm3D(128)                              │
│   ReLU(inplace=True)                            │
│   MaxPool3D(k=2, s=2)                           │
│   Dropout3D(p=0.1)                              │
│ Output: (B, 128, 12, 24, 24)                    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Block 4: Abstract Representations               │
│   Conv3D(128→256, k=3, s=1, p=1)                │
│   BatchNorm3D(256)                              │
│   ReLU(inplace=True)                            │
│   Dropout3D(p=0.1)                              │
│ Output: (B, 256, 12, 24, 24)                    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Global Pooling                                  │
│   AdaptiveAvgPool3D(output_size=1)              │
│ Output: (B, 256, 1, 1, 1)                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Classifier                                      │
│   Flatten → (B, 256)                            │
│   Linear(256 → 128)                             │
│   ReLU(inplace=True)                            │
│   Dropout(p=0.2)                                │
│   Linear(128 → 2)                               │
│ Output: (B, 2) [Control, Patient]               │
└─────────────────────────────────────────────────┘
```

### Design Decisions

#### 1. Input Channel Handling
**Decisión**: Eliminar canal b0, usar solo 64 canales de difusión

**Razón**: 
- Canal b0 (sin gradiente) aporta menos información direccional
- Reduce complejidad del modelo
- Enfoque en características de difusión pura

#### 2. Progressive Downsampling
**Estrategia**:
- Block 1: Mantener resolución completa (50×96×96)
- Blocks 2-3: MaxPool3D reduce a 12×24×24
- Block 4: Mantener resolución para features abstractos

**Razón**:
- Evita pérdida prematura de información espacial
- Permite aprender tanto features locales (block1) como globales (block4)

#### 3. Regularization Stack
**Técnicas aplicadas**:
1. **Dropout3D (0.1)** en bloques convolucionales → Previene co-adaptación de features
2. **BatchNorm3D** → Estabiliza entrenamiento, permite learning rates más altos
3. **Dropout (0.2)** antes de clasificador → Regularización fuerte en capas densas
4. **Weight Decay (1e-4)** en optimizer → Penalización L2 implícita

**Justificación**: Dataset pequeño (200 samples) requiere regularización agresiva

#### 4. Activation Functions
**Elección**: ReLU con `inplace=True`

**Razón**:
- ReLU: No sufre vanishing gradient, computacionalmente eficiente
- `inplace=True`: Ahorra memoria (crítico para volúmenes 3D grandes)

#### 5. Pooling Strategy
**AdaptiveAvgPool3D** vs MaxPool3D final:
- **MaxPool3D** en blocks intermedios → Selecciona features más activados
- **AdaptiveAvgPool3D** al final → Promedia información espacial completa

---

## 🎯 Training Strategy

### Loss Function: CrossEntropyLoss

Para clasificación binaria, PyTorch usa:

$$
\mathcal{L} = -\log\left(\frac{\exp(z_y)}{\sum_{c=1}^{C} \exp(z_c)}\right)
$$

Donde:
- $z_c$ = logit para clase $c$
- $y$ = clase verdadera

**Equivalente a**: Softmax + Negative Log-Likelihood

### Optimizer: Adam

**Parámetros**:
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)
```

**Adam update rule**:
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

### Early Stopping: Weighted Accuracy

**Métrica de selección**:
$$
\text{Weighted Acc} = 0.4 \times \text{Train Acc} + 0.6 \times \text{Val Acc}
$$

**Razón**:
- Penaliza overfitting (val tiene mayor peso)
- Premia generalización
- Mejor que usar solo val acc (puede ser ruidoso con pocas muestras)

---

## 📊 Evaluation Metrics

### Balanced Accuracy

Para datasets desbalanceados:

$$
\text{Balanced Acc} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}
$$

Donde $C$ = número de clases (2 en nuestro caso)

**Ventaja sobre accuracy estándar**:
- No favorece predicción de clase mayoritaria
- Útil cuando clases tienen prevalencias diferentes
- En nuestro dataset: 130 controls vs 132 patients (casi balanceado, pero importante en validación)

### Confusion Matrix Interpretation

```
                Predicted
              Control  Patient
Actual Control   TN       FP
       Patient   FN       TP
```

**Métricas derivadas**:
- **Sensitivity (Recall)**: $\frac{TP}{TP + FN}$ → % de pacientes correctamente identificados
- **Specificity**: $\frac{TN}{TN + FP}$ → % de controles correctamente identificados
- **Precision**: $\frac{TP}{TP + FP}$ → De los predichos como paciente, % que realmente lo son

**En contexto clínico**:
- Alta sensibilidad → No perder pacientes (crítico en screening)
- Alta especificidad → Evitar falsos positivos (reduce ansiedad y costos)

---

## 🔍 Interpretability Analysis

### t-SNE: Visualizing Learned Representations

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**:

Reduce embeddings de 256D (CNN) o 512D (SVM) a 2D preservando estructura local.

**Objetivo de optimización**:
$$
\min \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

Donde:
- $p_{ij}$ = similitud en espacio original (alta dimensión)
- $q_{ij}$ = similitud en espacio 2D

**Interpretación en nuestro proyecto**:
- **Clusters separados** → Modelo aprendió features discriminativas
- **Puntos mezclados** → Clases no son linealmente separables
- **Outliers** → Casos difíciles o mala calidad de datos

### Activation Maps: What Does the CNN See?

**Grad-CAM (futuro trabajo)** puede localizar regiones cerebrales que el modelo usa para clasificación:

$$
L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
$$

Donde:
- $\alpha_k^c = \frac{1}{Z} \sum_{i,j,k} \frac{\partial y^c}{\partial A_{ijk}^k}$ = importancia del filtro $k$ para clase $c$
- $A^k$ = activation map del filtro $k$

---

## 🚀 Performance Optimization

### Memory Management for 3D Volumes

**Problema**: Volumen completo = 96×96×50×64 × 4 bytes (float32) ≈ **113 MB por sujeto**

**Soluciones implementadas**:
1. **Batch size pequeño** (8) → Máximo 904 MB en GPU
2. **Dropout con inplace=True** → Reutiliza memoria
3. **Gradient checkpointing** (no implementado aún) → Trade-off memoria↔tiempo

### Data Loading Pipeline

```python
DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,     # Carga paralela
    pin_memory=True    # Acelera CPU→GPU transfer
)
```

**pin_memory=True**: Pre-aloja datos en RAM pinned → Transfer directo a GPU sin pasar por pageable memory (2-3× más rápido)



