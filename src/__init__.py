"""
TFM Sistema de Mantenimiento Predictivo
======================================

Sistema completo de mantenimiento predictivo para compresores industriales
desarrollado como Trabajo Fin de Máster (TFM) en Frío Pacífico 1, Concepción, Chile.

Módulos principales:
- tfm_pipeline: Pipeline principal de análisis y detección de anomalías
- monitor_sistema: Sistema de monitoreo continuo en tiempo real
- utils: Utilidades para datos y machine learning

Autor: Antonio
"""

__version__ = "1.0.0"
__author__ = "Antonio"
__email__ = "contacto@tfm-predictivo.cl"
__description__ = "Sistema de Mantenimiento Predictivo mediante Análisis THD"

# Importaciones principales
try:
    from .tfm_pipeline import TFMPipeline
    from .monitor_sistema import MonitorSistema
    from .utils.data_utils import DataLoader
    from .utils.ml_utils import TFMAnomalyDetector

    __all__ = [
        'TFMPipeline',
        'MonitorSistema', 
        'DataLoader',
        'TFMAnomalyDetector'
    ]
except ImportError:
    # Permitir importación parcial durante desarrollo
    __all__ = []
