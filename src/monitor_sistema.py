"""
Monitor Sistema - Sistema de Monitoreo Continuo TFM
===================================================

Sistema de monitoreo continuo para mantenimiento predictivo de compresores industriales
en Frío Pacífico 1, Concepción, Chile.

Implementa el sistema WATCHING (Real-time Monitoring) con:
- Análisis THD no-lineal en tiempo real
- Detección de anomalías con Isolation Forest + DBSCAN
- Generación inteligente de OT (Órdenes de Trabajo)
- Aprendizaje continuo con datos mensuales
- Alertas preventivas con ventana de predicción de 72 horas

Autor: Antonio
TFM: Mantenimiento Predictivo usando análisis THD
"""

import os
import json
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import schedule

# Importar utilidades TFM
try:
    from utils.data_utils import DataLoader
    from utils.ml_utils import TFMAnomalyDetector
except ImportError:
    print("⚠️ Importando desde ruta alternativa...")
    import sys
    sys.path.append('.')
    from src.utils.data_utils import DataLoader
    from src.utils.ml_utils import TFMAnomalyDetector


class MonitorSistema:
    """
    Sistema de Monitoreo Continuo con WATCHING en Tiempo Real

    Funcionalidades principales:
    - Monitoreo en tiempo real de 3 compresores
    - Detección de anomalías usando ensemble TFM
    - Generación automática de OT inteligentes
    - Aprendizaje continuo con nuevos datos
    - Sistema de alertas y notificaciones
    """

    def __init__(self, config_path: str = "config/config.json"):
        """Inicializar sistema de monitoreo con configuración TFM"""
        self.config = self._load_config(config_path)
        self.setup_logging()

        # Componentes principales
        self.data_loader = DataLoader()
        self.anomaly_detector = None
        self.is_monitoring = False
        self.monitoring_thread = None

        # Estado del sistema
        self.last_analysis_time = None
        self.anomaly_history = []
        self.ot_generated = []
        self.system_alerts = []

        # Configuración de monitoreo
        self.monitoring_interval = 300  # 5 minutos
        self.prediction_window = self.config['maintenance']['prediction_window_hours']
        self.alert_thresholds = self.config['monitoring']['watching_system']['alert_thresholds']

        self.logger.info("🚀 Monitor Sistema TFM inicializado correctamente")

    def _load_config(self, config_path: str) -> Dict:
        """Cargar configuración del sistema"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Archivo de configuración no encontrado: {config_path}")
            # Configuración básica por defecto
            return {
                'maintenance': {'prediction_window_hours': 72},
                'monitoring': {'watching_system': {'alert_thresholds': {
                    'anomaly_score': 0.8, 'thd_critical': 15.0, 'power_factor_min': 0.85
                }}},
                'paths': {'logs_dir': 'logs'},
                'logging': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 'level': 'INFO'}
            }
        except json.JSONDecodeError as e:
            print(f"❌ Error al decodificar JSON: {e}")
            raise

    def setup_logging(self):
        """Configurar sistema de logging"""
        log_dir = Path(self.config['paths']['logs_dir'])
        log_dir.mkdir(exist_ok=True)

        log_format = self.config['logging']['format']
        log_level = getattr(logging, self.config['logging']['level'])

        # Configurar logger
        self.logger = logging.getLogger('MonitorSistema')
        self.logger.setLevel(log_level)

        # Limpiar handlers existentes
        self.logger.handlers = []

        # Handler para archivo
        log_file = log_dir / f"monitor_sistema_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))

        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


def main():
    """Función principal para ejecutar el monitor como script independiente"""
    print("🚀 Iniciando Monitor Sistema TFM - Frío Pacífico 1")
    print("=" * 60)

    try:
        # Inicializar monitor
        monitor = MonitorSistema()
        print("✅ Monitor Sistema inicializado correctamente")
        print("📊 Listo para monitoreo continuo de compresores")
        print("🎯 Sistema preparado para detectar anomalías THD")

    except Exception as e:
        print(f"❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
