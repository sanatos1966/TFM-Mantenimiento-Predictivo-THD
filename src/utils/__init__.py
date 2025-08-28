"""
Utilidades TFM - Módulos de soporte
===================================

Contiene las utilidades principales para el sistema de mantenimiento predictivo:

- data_utils: Carga, limpieza y procesamiento de datos de sensores
- ml_utils: Modelos de machine learning y detección de anomalías

Estas utilidades implementan la metodología exacta del TFM con los parámetros
validados para reproducir los resultados de 439 anomalías y F1-Score de 0.963.
"""

try:
    from .data_utils import DataLoader
    from .ml_utils import TFMAnomalyDetector

    __all__ = ['DataLoader', 'TFMAnomalyDetector']
except ImportError:
    __all__ = []
