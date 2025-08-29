# Registros de Mantenimiento - GMAO Frío Pacífico 1

## Órdenes de Trabajo (OTs)

### Archivos incluidos:
- `ot_correctivas.csv`: 12 OTs correctivas (enero-julio 2025)
- `ot_preventivas.xlsx`: 47 OTs preventivas programadas  
- `gmao_export_completo.csv`: Export completo sistema GMAO
- `inspecciones_rutinarias.xlsx`: Inspecciones periódicas

### Validación TFM:
- **Total OTs procesadas**: 4,209 órdenes
- **Correctivos detectados**: 12/12 (100%)
- **Preventivos anticipados**: 34/47 (72.3%)
- **Correlación preventivos**: 83.1% precisión
- **MTTD alcanzado**: 69.8 horas
- **Período análisis**: Enero-julio 2025

### Estructura datos:
- ID_OT, Fecha_Creacion, Tipo_Mantenimiento, Compresor_ID
- Descripcion, Estado, Fecha_Cierre, Tiempo_Resolucion
- Prioridad, Tecnico_Asignado, Costo_Total

### Integración con algoritmos:
- Validación cruzada con anomalías IF+DBSCAN
- Correlación temporal con variables THD-vibraciones
- Base para cálculo métricas F1=0.963, AUC=0.981

