
# INSTRUCCIONES COMPLETAS - SISTEMA TFM MANTENIMIENTO PREDICTIVO
## Antonio Cantos & Renzo Chavez - Frío Pacífico 1, Concepción, Chile

INSTRUCCIONES COMPLETAS - SISTEMA TFM MANTENIMIENTO PREDICTIVO
Antonio Cantos & Renzo Chavez - Frío Pacífico 1, Concepción, Chile
🎯 OBJETIVO

Ejecutar y validar el sistema completo de mantenimiento predictivo que reproduce exactamente los resultados académicos del TFM:

101,646 registros industriales reales

439 anomalías detectadas

F1-Score = 0.963

Ensemble Isolation Forest + DBSCAN (70/30)

📁 ESTRUCTURA DEL REPOSITORIO

Tras clonar desde GitHub:

1. **validate_structure.py** - Script de validación de estructura del proyecto
2. **run_tfm_pipeline.py** - Pipeline completo del sistema TFM
   

=== VALIDACIÓN ESTRUCTURA PROYECTO TFM ===


TFM_pipeline/
config/
config.json
data/
raw/ -> Ficheros originales (.csv, .xlsx, .pdf)
processed/ -> Ficheros procesados
samples/ -> Datos de ejemplo
src/ -> Código fuente
tests/ -> Tests automáticos
output/ -> Resultados y reportes
run_tfm_pipeline.py -> Pipeline completo
watch_new_raw.py -> Script de monitoreo de nuevos datos
README.md -> Documentación principal


=== VALIDACIÓN COMPLETA ===


```
#### Paso 3: Ejecutar pipeline completo TFM
```powershell
# Ejecutar sistema completo
python run_tfm_pipeline.py
```
**Resultado esperado:**
```
```
#### Paso 4: Activar procesamiento automático (Watcher)

Además del pipeline completo, el sistema incluye un script de *watching* que permite
vigilar la carpeta `data/raw/` y procesar automáticamente nuevos archivos en formato
`.csv`, `.xlsx` o `.pdf`.

```powershell
python watch_new_raw.py
```
Comportamiento:
Vigila la carpeta data/raw/.
Procesa automáticamente cualquier archivo nuevo en formato .csv, .xlsx o .pdf.
Detecta anomalías en tiempo real con SistemaMantenimientoPredictivo.
Genera órdenes de trabajo con GeneradorOrdenesTrabajo.
Actualiza modelos con SistemaAprendizajeContinuo.
Guarda un registro de los archivos ya procesados en output/_processed_files.json.
Nota: El intervalo de revisión por defecto es de 5 minutos. Puedes modificarlo editando la variable SLEEP_SEC en watch_new_raw.py.
```
==================================================
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


### 📞 CONTACTO Y SOPORTE
Si encuentras algún problema durante la ejecución, documenta:
1. Mensaje de error exacto
2. Paso donde ocurre el problema
3. Versión de Python (python --version)

### 🎓 RECONOCIMIENTOS ACADÉMICOS
Sistema desarrollado para TFM de Mantenimiento Predictivo Industrial
- **Institución**: [EADIC 2025]
- **Ubicación**: Frío Pacífico 1, Concepción, Chile
- **Tecnología**: Ensemble Machine Learning (Isolation Forest + DBSCAN)
- **Datos Reales**: 182,670 registros industriales de compresores
