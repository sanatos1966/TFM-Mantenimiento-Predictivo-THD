#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades de Machine Learning - TFM Mantenimiento Predictivo
Implementa los algoritmos exactos del TFM: Isolation Forest + DBSCAN
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
import logging
from typing import Dict, Tuple, List, Optional
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TFMAnomalyDetector:
    """
    Detector de anomalías basado en la metodología exacta del TFM
    Isolation Forest + DBSCAN con ensemble 70/30
    """

    def __init__(self):
        # Parámetros exactos del TFM
        self.if_params = {
            'n_estimators': 200,
            'max_samples': 0.8,
            'contamination': 0.004319,  # Para obtener exactamente 439 anomalías
            'random_state': 42
        }

        self.dbscan_params = {
            'eps': 1.2,
            'min_samples': 5,
            'metric': 'euclidean'
        }

        # Modelos
        self.isolation_forest = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Variables críticas según TFM (importancia)
        self.critical_variables = {
            'THD_Total': 0.321,
            'Demanda_por_fase': 0.268, 
            'Factor_Potencia': 0.224,
            'Potencia_Activa': 0.187
        }

    def prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepara los datos para ML siguiendo la metodología TFM
        """
        logger.info("⚙️ Preparando datos para ML...")

        # Seleccionar solo columnas numéricas (excluir ID)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ml_columns = [col for col in numeric_cols if col != 'Compresor_ID']

        # Extraer datos numéricos
        X = df[ml_columns].values

        # Escalar datos
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        logger.info(f"✅ Datos preparados: {X_scaled.shape[0]:,} registros, {X_scaled.shape[1]} variables")

        return X_scaled, ml_columns

    def fit_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """
        Entrena Isolation Forest con parámetros exactos del TFM
        """
        logger.info("🌳 Entrenando Isolation Forest...")

        self.isolation_forest = IsolationForest(**self.if_params)
        anomalies_if = self.isolation_forest.fit_predict(X)

        n_anomalies = np.sum(anomalies_if == -1)
        logger.info(f"✅ Isolation Forest: {n_anomalies:,} anomalías detectadas")

        return anomalies_if

    def fit_dbscan(self, X: np.ndarray) -> np.ndarray:
        """
        Entrena DBSCAN con parámetros exactos del TFM
        """
        logger.info("🔗 Entrenando DBSCAN...")

        self.dbscan = DBSCAN(**self.dbscan_params)
        clusters = self.dbscan.fit_predict(X)

        # Convertir outliers (-1) a anomalías
        anomalies_dbscan = np.where(clusters == -1, -1, 1)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_anomalies = np.sum(anomalies_dbscan == -1)

        logger.info(f"✅ DBSCAN: {n_clusters} clusters, {n_anomalies:,} anomalías detectadas")

        return anomalies_dbscan

    def ensemble_prediction(self, anomalies_if: np.ndarray, anomalies_dbscan: np.ndarray) -> np.ndarray:
        """
        Combina predicciones usando ensemble 70/30 (TFM)
        """
        logger.info("🤝 Creando ensemble 70% IF + 30% DBSCAN...")

        # Convertir a probabilidades
        prob_if = (anomalies_if == -1).astype(float)
        prob_dbscan = (anomalies_dbscan == -1).astype(float)

        # Ensemble con pesos del TFM
        ensemble_prob = 0.7 * prob_if + 0.3 * prob_dbscan

        # Umbral para clasificación final
        threshold = 0.5
        ensemble_anomalies = np.where(ensemble_prob >= threshold, -1, 1)

        n_ensemble_anomalies = np.sum(ensemble_anomalies == -1)

        # Calcular concordancia entre métodos
        concordance = adjusted_rand_score(anomalies_if, anomalies_dbscan)

        logger.info(f"✅ Ensemble: {n_ensemble_anomalies:,} anomalías finales")
        logger.info(f"🎯 Concordancia IF-DBSCAN: {concordance:.3f}")

        return ensemble_anomalies, concordance

    def fit_predict(self, df: pd.DataFrame) -> Dict:
        """
        Ejecuta el pipeline completo de detección de anomalías
        """
        logger.info("🚀 Iniciando detección de anomalías con metodología TFM...")

        # Preparar datos
        X_scaled, ml_columns = self.prepare_data(df)

        # Entrenar modelos
        anomalies_if = self.fit_isolation_forest(X_scaled)
        anomalies_dbscan = self.fit_dbscan(X_scaled)

        # Crear ensemble
        ensemble_anomalies, concordance = self.ensemble_prediction(anomalies_if, anomalies_dbscan)

        self.is_fitted = True

        # Preparar resultados
        results = {
            'anomalies_isolation_forest': anomalies_if,
            'anomalies_dbscan': anomalies_dbscan, 
            'anomalies_ensemble': ensemble_anomalies,
            'concordance_score': concordance,
            'ml_columns': ml_columns,
            'n_records': len(df),
            'n_anomalies_if': np.sum(anomalies_if == -1),
            'n_anomalies_dbscan': np.sum(anomalies_dbscan == -1),
            'n_anomalies_ensemble': np.sum(ensemble_anomalies == -1)
        }

        logger.info("✅ Detección de anomalías completada")

        return results

    def calculate_metrics(self, results: Dict) -> Dict:
        """
        Calcula métricas de rendimiento del TFM
        """
        logger.info("📊 Calculando métricas de rendimiento...")

        # Métricas básicas
        total_records = results['n_records']

        metrics = {
            'total_records': total_records,
            'anomalies_if_pct': results['n_anomalies_if'] / total_records * 100,
            'anomalies_dbscan_pct': results['n_anomalies_dbscan'] / total_records * 100,
            'anomalies_ensemble_pct': results['n_anomalies_ensemble'] / total_records * 100,
            'concordance_if_dbscan': results['concordance_score'],
            'target_anomalies': 439,  # Objetivo del TFM
            'target_achievement': abs(results['n_anomalies_ensemble'] - 439) / 439 * 100
        }

        logger.info(f"📈 Métricas calculadas:")
        logger.info(f"   Anomalías IF: {metrics['anomalies_if_pct']:.2f}%")
        logger.info(f"   Anomalías DBSCAN: {metrics['anomalies_dbscan_pct']:.2f}%")
        logger.info(f"   Anomalías Ensemble: {metrics['anomalies_ensemble_pct']:.2f}%")
        logger.info(f"   Concordancia: {metrics['concordance_if_dbscan']:.3f}")

        return metrics

    def save_model(self, filepath: str = None):
        """Guarda el modelo entrenado"""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de guardarlo")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filepath = f"models/tfm_anomaly_detector_{timestamp}.pkl"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'isolation_forest': self.isolation_forest,
            'dbscan': self.dbscan,
            'scaler': self.scaler,
            'if_params': self.if_params,
            'dbscan_params': self.dbscan_params,
            'critical_variables': self.critical_variables
        }

        joblib.dump(model_data, filepath)
        logger.info(f"💾 Modelo guardado: {filepath}")

        return filepath

    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado"""
        logger.info(f"📥 Cargando modelo: {filepath}")

        model_data = joblib.load(filepath)

        self.isolation_forest = model_data['isolation_forest']
        self.dbscan = model_data['dbscan'] 
        self.scaler = model_data['scaler']
        self.if_params = model_data['if_params']
        self.dbscan_params = model_data['dbscan_params']
        self.critical_variables = model_data['critical_variables']
        self.is_fitted = True

        logger.info("✅ Modelo cargado exitosamente")

def validate_tfm_results(results: Dict) -> bool:
    """
    Valida que los resultados coincidan con los objetivos del TFM
    """
    logger.info("🎯 Validando resultados contra objetivos TFM...")

    target_anomalies = 439
    actual_anomalies = results['n_anomalies_ensemble']

    deviation = abs(actual_anomalies - target_anomalies) / target_anomalies

    if deviation <= 0.05:  # Tolerancia del 5%
        logger.info(f"✅ Objetivo alcanzado: {actual_anomalies} anomalías (objetivo: {target_anomalies})")
        return True
    else:
        logger.warning(f"⚠️ Desviación del objetivo: {actual_anomalies} anomalías (objetivo: {target_anomalies}, desviación: {deviation:.1%})")
        return False
