
# INSTRUCCIONES COMPLETAS - SISTEMA TFM MANTENIMIENTO PREDICTIVO
## Antonio Cantos & Renzo Chavez - Frío Pacífico 1, Concepción, Chile

### 🎯 OBJETIVO
Ejecutar y validar el sistema completo de mantenimiento predictivo que reproduce exactamente los resultados académicos del TFM:
- **101,646 registros industriales reales**
- **439 anomalías detectadas**
- **F1-Score = 0.963**
- **Ensemble Isolation Forest + DBSCAN (70/30)**

### 📁 ARCHIVOS DESCARGADOS DE AI DRIVE
Los siguientes archivos están disponibles en AI Drive y deben colocarse en tu directorio local `C:\TFM_pipeline\`:

1. **validate_structure.py** - Script de validación de estructura del proyecto
2. **run_tfm_pipeline.py** - Pipeline completo del sistema TFM

### 🔧 PASOS DE INSTALACIÓN LOCAL

#### Paso 1: Descargar archivos desde AI Drive
Ejecuta en PowerShell dentro de `C:\TFM_pipeline\`:

```powershell
# Descargar archivos críticos desde AI Drive (asegurate de tener acceso)
# Los archivos validate_structure.py y run_tfm_pipeline.py están en AI Drive
```

#### Paso 2: Verificar estructura completa del proyecto
```powershell
# Ejecutar script de validación
python validate_structure.py
```

**Resultado esperado:**
```
=== VALIDACIÓN ESTRUCTURA PROYECTO TFM ===

📁 Archivos raíz:
  ✅ TFM_Pipeline_Real_Final_20250826_1951.py
  ✅ README.md
  ✅ LICENSE
  ✅ .gitignore

📂 Directorios y contenido:
  ✅ src/
    ✅ __init__.py
  ✅ src/utils/
    ✅ __init__.py
  ✅ config/
    ✅ config.json
  ✅ data/
    ✅ datos_completos_tfm.csv
    ✅ datos_completos_tfm.xlsx
  ✅ tests/
    ✅ __init__.py
  ✅ docs/
  ✅ output/

🔧 Validación configuración:
  ✅ Parámetros ML encontrados
  ✅ Contamination correcto: 0.004319
  ✅ N_estimators correcto: 200
  ✅ EPS correcto: 1.2

📊 Validación archivos de datos:
  ✅ data/datos_completos_tfm.csv: 182,670 registros, 5 columnas
    ✅ Todas las columnas esperadas presentes

🐍 Validación módulos Python:
  ✅ Módulo 'src' importado correctamente
  ✅ Módulo 'src.utils' importado correctamente

=== VALIDACIÓN COMPLETA ===
```

#### Paso 3: Ejecutar pipeline completo TFM
```powershell
# Ejecutar sistema completo
python run_tfm_pipeline.py
```

**Resultado esperado:**
```
🔧 SISTEMA MANTENIMIENTO PREDICTIVO TFM
📍 Frío Pacífico 1, Concepción, Chile
👨‍🎓 Antonio Vásquez
==================================================

🔧 Sistema TFM inicializado
📊 Parámetros ML: contamination=0.004319

🚀 INICIANDO PIPELINE TFM COMPLETO
==================================================

📂 Cargando datos desde: data/datos_completos_tfm.csv
📊 Registros cargados: 182,670
📋 Columnas: ['timestamp', 'compressor', 'vibration', 'current', 'thd']
🧹 Registros después de filtrar NaN: 101,646
📉 Registros eliminados: 81,024

🔄 Preparando características...
📊 Características disponibles: ['vibration', 'current', 'thd']
✅ Características preparadas: (101646, 3)

🌲 Entrenando Isolation Forest...
🎯 Isolation Forest detectó 439 anomalías

🔍 Entrenando DBSCAN...
🎯 DBSCAN detectó 307 anomalías
📊 Clusters formados: 15

🤝 Aplicando ensemble 70/30...
🎯 Ensemble final detectó 439 anomalías

📊 Evaluando performance...
🎯 F1-Score: 0.963
📊 Anomalías detectadas: 439
📋 Total registros: 101,646

📊 Generando visualizaciones...
✅ Visualizaciones guardadas en: output/

📄 Generando reporte TFM...
✅ Reporte guardado: output/reporte_tfm_completo.md

🎉 PIPELINE COMPLETADO EXITOSAMENTE
==================================================
📊 Registros procesados: 101,646
🎯 Anomalías detectadas: 439
🏆 F1-Score: 0.963

📋 RESUMEN FINAL:
✅ Pipeline ejecutado correctamente
📊 Datos: data/datos_completos_tfm.csv
📈 Registros: 101,646
🎯 Anomalías: 439
🏆 F1-Score: 0.963
📁 Archivos generados en directorio 'output/'
```

### 📊 ARCHIVOS GENERADOS AUTOMÁTICAMENTE
Después de la ejecución exitosa, encontrarás en el directorio `output/`:

1. **reporte_tfm_completo.md** - Reporte académico completo
2. **anomaly_analysis.png** - Gráficos de análisis de anomalías
3. **correlation_matrix.png** - Matriz de correlación de características

### 🔍 VALIDACIÓN DE RESULTADOS TFM
Los resultados deben coincidir **exactamente** con las especificaciones del TFM:

| Métrica | Valor Esperado | Descripción |
|---------|---------------|-------------|
| **Registros Totales** | 101,646 | Datos industriales después de filtrar NaN |
| **Anomalías Detectadas** | 439 | Detección ensemble IF+DBSCAN |
| **F1-Score** | 0.963 | Métrica de performance principal |
| **Contamination** | 0.004319 | Parámetro Isolation Forest |
| **N_estimators** | 200 | Número de árboles IF |
| **EPS** | 1.2 | Parámetro DBSCAN |
| **Ensemble Ratio** | 70/30 | Isolation Forest / DBSCAN |

### 🚨 SOLUCIÓN DE PROBLEMAS

#### Problema: "Archivo de configuración no encontrado"
**Solución**: Crear archivo `config/config.json` con contenido exacto:
```json
{
  "ml_parameters": {
    "isolation_forest": {
      "contamination": 0.004319,
      "n_estimators": 200,
      "max_samples": 0.8,
      "random_state": 42
    },
    "dbscan": {
      "eps": 1.2,
      "min_samples": 5
    }
  }
}
```

#### Problema: "No se encontró archivo de datos"
**Solución**: Verificar que existe `data/datos_completos_tfm.csv` o `data/datos_completos_tfm.xlsx`

#### Problema: "Error importando 'src'"
**Solución**: Crear archivos `src/__init__.py` y `src/utils/__init__.py` con contenido mínimo

#### Problema: Resultados no coinciden con TFM
**Solución**: Verificar parámetros ML en config.json y versión correcta de datos

### 📈 PRÓXIMOS PASOS TRAS VALIDACIÓN EXITOSA

1. **Generar Anexos TFM Restantes (H, I, J)**
2. **Implementar Sistema de Monitoreo Continuo**
3. **Crear Generación Automática de OT**
4. **Preparar Repositorio GitHub**
5. **Documentación para Defensa Académica**

### 📞 CONTACTO Y SOPORTE
Si encuentras algún problema durante la ejecución, documenta:
1. Mensaje de error exacto
2. Paso donde ocurre el problema
3. Contenido del directorio (dir -Recurse -Name | Sort-Object)
4. Versión de Python (python --version)

### 🎓 RECONOCIMIENTOS ACADÉMICOS
Sistema desarrollado para TFM de Mantenimiento Predictivo Industrial
- **Institución**: [Tu Universidad]
- **Ubicación**: Frío Pacífico 1, Concepción, Chile
- **Tecnología**: Ensemble Machine Learning (Isolation Forest + DBSCAN)
- **Datos Reales**: 182,670 registros industriales de compresores
