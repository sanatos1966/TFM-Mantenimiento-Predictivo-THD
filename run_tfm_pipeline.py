"""
Sistema de Mantenimiento Predictivo TFM - Antonio Cantos & Renzo Chavez
Frío Pacífico 1, Concepción, Chile

Pipeline completo para reproducir resultados académicos exactos:
- 101,646 registros industriales reales
- 439 anomalías detectadas
- F1-Score = 0.963
- Ensemble: Isolation Forest + DBSCAN (70/30)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TFMPredictiveMaintenanceSystem:
    """
    Sistema de Mantenimiento Predictivo TFM
    Implementa ensemble Isolation Forest + DBSCAN para detección de anomalías
    """

    def __init__(self, config_path='config/config.json'):
        """Inicializar sistema con configuración TFM"""
        self.config = self.load_config(config_path)
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.dbscan = None
        self.results = {}

        if_cfg = self.config['models']['isolation_forest']
        print("🔧 Sistema TFM inicializado")
        print(f"📊 Parámetros IF: contamination={if_cfg['contamination']}, "
              f"n_estimators={if_cfg['n_estimators']}, max_samples={if_cfg['max_samples']}")

    def load_config(self, config_path):
        """Cargar configuración del sistema"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_data(self, file_path):
        """
        Cargar datos industriales reales
        Soporta CSV y Excel con filtrado automático de NaN
        """
        print(f"📂 Cargando datos desde: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo de datos no encontrado: {file_path}")

        # Cargar según extensión
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine="openpyxl")  # <- forzar openpyxl
        else:
            raise ValueError("Formato de archivo no soportado. Use CSV o Excel (.xlsx).")

        # Información inicial
        print(f"📊 Registros cargados: {len(df):,}")
        print(f"📋 Columnas: {list(df.columns)}")

        # Filtrar NaN
        initial_count = len(df)
        df = df.dropna()
        final_count = len(df)
        print(f"🧹 Registros después de filtrar NaN: {final_count:,} "
              f"(eliminados: {initial_count - final_count:,})")

        # Validar columnas esperadas (ajústalas a tu dataset real)
        expected_columns = ['timestamp', 'compressor', 'vibration', 'current', 'thd']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️  Columnas faltantes (no bloqueante): {missing_columns}")

        # Convertir timestamp si existe
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])

        self.data = df
        return df

    def prepare_features(self, df):
        """
        Preparar características para análisis ML (normalización estándar)
        """
        print("🔄 Preparando características...")
        feature_columns = ['vibration', 'current', 'thd']
        available_features = [c for c in feature_columns if c in df.columns]
        print(f"📊 Características disponibles: {available_features}")
        if not available_features:
            raise ValueError("No se encontraron características válidas para ML")

        X = df[available_features].copy()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns[... if necessary.
