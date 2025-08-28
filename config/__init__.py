"""
Configuración TFM
=================

Módulo de configuración para el sistema de mantenimiento predictivo.
Contiene los parámetros exactos validados en el TFM.
"""

import json
import os
from pathlib import Path

def load_config(config_path="config/config.json"):
    """Cargar configuración del sistema TFM"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tfm_parameters():
    """Obtener parámetros exactos del TFM validados"""
    return {
        'total_records': 101646,
        'expected_anomalies': 439,
        'target_f1_score': 0.963,
        'isolation_forest': {
            'n_estimators': 200,
            'max_samples': 0.8,
            'contamination': 0.004319,
            'random_state': 42
        },
        'dbscan': {
            'eps': 1.2,
            'min_samples': 5
        },
        'ensemble_weights': {
            'isolation_forest': 0.7,
            'dbscan': 0.3
        },
        'critical_variables': {
            'THD_Total': 32.1,
            'Demanda_por_fase': 26.8,
            'Factor_Potencia': 22.4,
            'Potencia_Activa': 18.7
        }
    }
