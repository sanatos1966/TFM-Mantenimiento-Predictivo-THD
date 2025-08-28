#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades para manejo de datos - TFM Mantenimiento Predictivo
Autor: Antonio Cantos % Renzo Chavez - Sistema Frío Pacífico 1
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataLoader:
    """Cargador de datos para el sistema de mantenimiento predictivo"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"

    def load_compressor_data(self) -> pd.DataFrame:
        """
        Carga y combina los datos de los 3 compresores
        Reproduce exactamente los 101,646 registros del TFM
        """
        logger.info("🔄 Cargando datos de compresores...")

        # Rutas de archivos de compresores
        sensor_path = self.data_path / "raw" / "sensor"
        compressor_files = {
            'C1': sensor_path / "Compresor1_FP1.csv",
            'C2': sensor_path / "Compresor2_FP1.csv", 
            'C3': sensor_path / "Compresor3_FP1.csv"
        }

        combined_data = []

        for comp_id, file_path in compressor_files.items():
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    df['Compresor_ID'] = comp_id
                    combined_data.append(df)
                    logger.info(f"✅ {comp_id}: {len(df):,} registros cargados")
                except Exception as e:
                    logger.error(f"❌ Error cargando {comp_id}: {str(e)}")
            else:
                logger.warning(f"⚠️ Archivo no encontrado: {file_path}")

        if not combined_data:
            raise FileNotFoundError("No se pudieron cargar archivos de compresores")

        # Combinar todos los datos
        df_combined = pd.concat(combined_data, ignore_index=True)

        logger.info(f"✅ Dataset combinado: {len(df_combined):,} registros totales")

        return df_combined

    def load_maintenance_records(self) -> pd.DataFrame:
        """Carga las órdenes de trabajo (OT)"""
        logger.info("🔄 Cargando órdenes de trabajo...")

        ot_path = self.data_path / "raw" / "maintenance_records" / "OT compresores.csv"

        if not ot_path.exists():
            raise FileNotFoundError(f"Archivo de OT no encontrado: {ot_path}")

        try:
            df_ot = pd.read_csv(ot_path, encoding='utf-8')
            logger.info(f"✅ {len(df_ot):,} órdenes de trabajo cargadas")
            return df_ot
        except Exception as e:
            logger.error(f"❌ Error cargando OT: {str(e)}")
            raise

    def clean_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos de sensores siguiendo la metodología del TFM
        """
        logger.info("🧹 Limpiando datos de sensores...")

        initial_records = len(df)
        initial_columns = len(df.columns)

        # Eliminar columnas completamente vacías
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logger.info(f"🗑️ Eliminadas {len(empty_cols)} columnas vacías")

        # Eliminar columnas con más del 95% de NaN
        high_nan_cols = []
        for col in df.columns:
            if col != 'Compresor_ID' and df[col].isnull().sum() / len(df) > 0.95:
                high_nan_cols.append(col)

        if high_nan_cols:
            df = df.drop(columns=high_nan_cols)
            logger.info(f"🗑️ Eliminadas {len(high_nan_cols)} columnas con >95% NaN")

        # Eliminar filas con demasiados NaN
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)

        # Rellenar NaN restantes con mediana (solo columnas numéricas)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Compresor_ID' and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        final_records = len(df)
        final_columns = len(df.columns)

        logger.info(f"✅ Limpieza completada:")
        logger.info(f"   Registros: {initial_records:,} → {final_records:,}")
        logger.info(f"   Columnas: {initial_columns} → {final_columns}")

        return df

    def validate_tfm_structure(self, df: pd.DataFrame) -> bool:
        """
        Valida que el dataset tenga la estructura esperada del TFM
        """
        logger.info("🔍 Validando estructura del dataset...")

        # Verificar compresores
        expected_compressors = ['C1', 'C2', 'C3']
        actual_compressors = sorted(df['Compresor_ID'].unique())

        if actual_compressors != expected_compressors:
            logger.error(f"❌ Compresores esperados: {expected_compressors}, encontrados: {actual_compressors}")
            return False

        # Verificar distribución aproximada (según TFM: C1:53.8%, C2:23.2%, C3:23.0%)
        distribution = df['Compresor_ID'].value_counts(normalize=True)

        expected_dist = {'C1': 0.538, 'C2': 0.232, 'C3': 0.230}
        for comp, expected_pct in expected_dist.items():
            actual_pct = distribution.get(comp, 0)
            deviation = abs(actual_pct - expected_pct)

            if deviation > 0.1:  # Tolerancia del 10%
                logger.warning(f"⚠️ {comp}: distribución esperada {expected_pct:.1%}, actual {actual_pct:.1%}")

        logger.info("✅ Estructura del dataset validada")
        return True

def save_processed_data(df: pd.DataFrame, output_path: str = None) -> str:
    """Guarda el dataset procesado"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"data/processed/compresores_combinados_{timestamp}.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"💾 Dataset guardado: {output_path}")

    return output_path

def load_new_monthly_data(file_path: str) -> pd.DataFrame:
    """
    Carga nuevos datos mensuales (CSV, Excel, o procesados desde PDF)
    """
    logger.info(f"🔄 Cargando nuevos datos: {file_path}")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Formato no soportado: {file_path.suffix}")

        logger.info(f"✅ Nuevos datos cargados: {len(df):,} registros")
        return df

    except Exception as e:
        logger.error(f"❌ Error cargando datos: {str(e)}")
        raise
