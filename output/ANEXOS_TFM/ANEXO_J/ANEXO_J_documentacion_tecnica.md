# ANEXO J - CÓDIGO FUENTE Y DOCUMENTACIÓN

## 1. Arquitectura del Sistema

### 1.1 Componentes Principales
- **SistemaMantenimientoPredictivo**: Clase principal
- **DataProcessor**: Procesamiento de datos
- **ModeloEnsemble**: Isolation Forest + DBSCAN
- **IntegradorGMAO**: Integración con sistema GMAO

### 1.2 Estructura de Archivos
```
TFM_Pipeline/
├── src/
│   ├── tfm_pipeline_main.py      # Sistema principal
│   ├── data_processor.py         # Procesamiento datos
│   ├── modelo_ensemble.py        # Modelos ML
│   └── integrador_gmao.py        # Integración GMAO
├── config/
│   └── config.json               # Configuración
├── data/
│   ├── raw/                      # Datos originales
│   └── processed/                # Datos procesados
└── output/
    ├── models/                   # Modelos entrenados
    └── reports/                  # Reportes generados
```

## 2. Instalación y Uso

### 2.1 Requisitos
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

### 2.2 Ejecución
```python
from tfm_pipeline_main import SistemaMantenimientoPredictivo

# Crear instancia
sistema = SistemaMantenimientoPredictivo()

# Entrenar modelo
sistema.entrenar_modelo(X_train)

# Detectar anomalías
anomalias = sistema.detectar_anomalias(X_new)
```

## 3. Configuración

### 3.1 Parámetros del Modelo
- **Isolation Forest**: contamination=0.1, n_estimators=100
- **DBSCAN**: eps=0.5, min_samples=5
- **Ensemble**: OR lógico entre predicciones

### 3.2 Variables de Entrada
- THD (Distorsión Armónica Total)
- Factor de Potencia
- Potencia Activa
- Presión de Descarga
- Temperatura

## 4. API y Endpoints

### 4.1 Endpoints Principales
- `POST /predict`: Detectar anomalías
- `GET /status`: Estado del sistema
- `POST /retrain`: Reentrenar modelo
- `GET /metrics`: Métricas de rendimiento

---
*Fuente: Documentación técnica TFM*
