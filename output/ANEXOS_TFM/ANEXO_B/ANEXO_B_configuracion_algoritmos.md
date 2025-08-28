# ANEXO B: CONFIGURACIÓN DE ALGORITMOS

## Modelo Ensemble Implementado

El sistema utiliza un enfoque ensemble combinando dos algoritmos:

### 1. Isolation Forest

- **Contaminación**: 0.11351653225144134
- **Número de estimadores**: 200
- **Semilla aleatoria**: 42

### 2. DBSCAN

- **Epsilon**: 0.5
- **Mínimo de muestras**: 10

### 3. Variables Utilizadas

- THD
- Potencia_Activa

### 4. Estrategia de Ensemble

- **Combinación**: OR lógico (anomalía si cualquier modelo la detecta)
- **Ventaja**: Mayor sensibilidad para detectar diferentes tipos de anomalías
