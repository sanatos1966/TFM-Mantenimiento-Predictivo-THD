# Reportes TFM - Mantenimiento Predictivo

## Estructura de Reportes:

### 📊 anomaly_reports/
- Reportes detallados de las 439 anomalías detectadas
- Análisis por compresor (C1: 54%, C2: 23%, C3: 23%)
- Timeline de eventos anómalos enero-julio 2025
- Correlación con órdenes de trabajo GMAO

### 📈 performance_metrics/
- Métricas del modelo ensemble IF+DBSCAN
- F1-Score: 0.963, AUC: 0.981, Precisión: 94.7%
- Curvas ROC y matrices de confusión
- Comparativa algoritmos individuales vs ensemble

### 🔗 correlation_analysis/
- Análisis correlaciones THD-vibraciones (r>0.73)
- 4 correlaciones específicas del estudio
- Coherencia espectral electro-mecánica
- Evolución temporal de dependencias

### 📋 executive_summaries/
- Resúmenes ejecutivos por período
- KPIs de mantenimiento predictivo
- Impacto económico y ROI (42% reducción MTBF)
- Recomendaciones operacionales

## Formatos incluidos:
- PDF: Reportes formales y presentaciones
- HTML: Dashboards interactivos
- CSV: Datos tabulares exportables
- PNG: Gráficos de alta calidad

## Automatización:
- Generación automática cada procesamiento
- Integración con pipeline principal
- Exportación directa desde Jupyter notebooks
- Sincronización con sistema GMAO

