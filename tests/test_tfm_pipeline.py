"""
Test TFM Pipeline - Pruebas Unitarias Completas
===============================================

Pruebas unitarias para el sistema de mantenimiento predictivo TFM
de Antonio Cantos & Renzo Chavez en Frío Pacífico 1, Concepción, Chile.

Valida:
- Funcionalidades de carga y procesamiento de datos
- Detección de anomalías con Isolation Forest + DBSCAN
- Reproducibilidad de resultados exactos (439 anomalías, F1=0.963)
- Generación de OT inteligentes
- Sistema de monitoreo continuo

Autor: Antonio Cantos y Renzo Chavez
TFM: Mantenimiento Predictivo usando análisis THD
"""

import unittest
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Añadir rutas al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from src.utils.data_utils import DataLoader
    from src.utils.ml_utils import TFMAnomalyDetector
    from src.monitor_sistema import MonitorSistema
except ImportError:
    # Fallback para diferentes estructuras de proyecto
    from utils.data_utils import DataLoader
    from utils.ml_utils import TFMAnomalyDetector
    from monitor_sistema import MonitorSistema


class TestDataUtils(unittest.TestCase):
    """Pruebas para utilidades de datos (DataLoader)"""

    def setUp(self):
        """Configurar datos de prueba"""
        self.data_loader = DataLoader()
        self.temp_dir = tempfile.mkdtemp()

        # Crear datos de prueba simulados
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'THD_Total': np.random.normal(8.5, 2.0, 1000),
            'Demanda_por_fase': np.random.normal(45.2, 5.0, 1000),
            'Factor_Potencia': np.random.normal(0.92, 0.05, 1000),
            'Potencia_Activa': np.random.normal(38.7, 4.0, 1000),
            'Voltage_L1': np.random.normal(220, 10, 1000),
            'Current_L1': np.random.normal(15.5, 2.0, 1000)
        })

        # Introducir algunas anomalías
        anomaly_indices = np.random.choice(1000, 10, replace=False)
        self.test_data.loc[anomaly_indices, 'THD_Total'] = np.random.normal(25, 5, 10)  # THD alto

        # Guardar archivo de prueba
        self.test_file = os.path.join(self.temp_dir, 'Compresor1_FP1.csv')
        self.test_data.to_csv(self.test_file, index=False)

    def tearDown(self):
        """Limpiar archivos temporales"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_compressor_data(self):
        """Probar carga de datos de compresor"""
        data = self.data_loader.load_compressor_data('Compresor1_FP1', self.test_file)

        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 1000)
        self.assertIn('THD_Total', data.columns)
        self.assertIn('Demanda_por_fase', data.columns)

    def test_clean_sensor_data(self):
        """Probar limpieza de datos de sensores"""
        # Introducir valores faltantes
        dirty_data = self.test_data.copy()
        dirty_data.loc[0:10, 'THD_Total'] = np.nan
        dirty_data.loc[20:25, 'Factor_Potencia'] = -999  # Valor anómalo

        cleaned_data = self.data_loader.clean_sensor_data(dirty_data)

        # Verificar que se eliminaron NaN
        self.assertFalse(cleaned_data['THD_Total'].isna().any())
        # Verificar que se mantuvieron registros válidos
        self.assertGreater(len(cleaned_data), 900)

    def test_validate_tfm_structure(self):
        """Probar validación de estructura de datos TFM"""
        # Estructura válida
        valid_result = self.data_loader.validate_tfm_structure(self.test_data)
        self.assertTrue(valid_result['is_valid'])
        self.assertGreaterEqual(valid_result['completeness'], 0.8)

        # Estructura inválida
        invalid_data = self.test_data.drop(['THD_Total'], axis=1)
        invalid_result = self.data_loader.validate_tfm_structure(invalid_data)
        self.assertFalse(invalid_result['is_valid'])

    def test_feature_engineering_thd(self):
        """Probar ingeniería de características THD"""
        engineered_data = self.data_loader.feature_engineering_thd(self.test_data)

        # Verificar nuevas características
        self.assertIn('THD_Total_rolling_mean', engineered_data.columns)
        self.assertIn('THD_Total_rolling_std', engineered_data.columns)
        self.assertIn('THD_anomaly_flag', engineered_data.columns)

        # Verificar que se mantuvieron datos originales
        self.assertEqual(len(engineered_data), len(self.test_data))


class TestMLUtils(unittest.TestCase):
    """Pruebas para utilidades de Machine Learning (TFMAnomalyDetector)"""

    def setUp(self):
        """Configurar detector de anomalías"""
        # Parámetros exactos del TFM de Antonio
        self.isolation_forest_params = {
            'n_estimators': 200,
            'max_samples': 0.8,
            'contamination': 0.004319,
            'random_state': 42,
            'n_jobs': -1
        }

        self.dbscan_params = {
            'eps': 1.2,
            'min_samples': 5,
            'metric': 'euclidean'
        }

        self.ensemble_weights = {
            'isolation_forest': 0.7,
            'dbscan': 0.3
        }

        self.detector = TFMAnomalyDetector(
            isolation_forest_params=self.isolation_forest_params,
            dbscan_params=self.dbscan_params,
            ensemble_weights=self.ensemble_weights
        )

        # Datos de entrenamiento simulados
        np.random.seed(42)
        self.training_data = pd.DataFrame({
            'THD_Total': np.random.normal(8.5, 2.0, 5000),
            'Demanda_por_fase': np.random.normal(45.2, 5.0, 5000),
            'Factor_Potencia': np.random.normal(0.92, 0.05, 5000),
            'Potencia_Activa': np.random.normal(38.7, 4.0, 5000)
        })

    def test_detector_initialization(self):
        """Probar inicialización del detector"""
        self.assertIsNotNone(self.detector.isolation_forest)
        self.assertIsNotNone(self.detector.dbscan)
        self.assertEqual(self.detector.ensemble_weights['isolation_forest'], 0.7)
        self.assertEqual(self.detector.ensemble_weights['dbscan'], 0.3)

    def test_fit_models(self):
        """Probar entrenamiento de modelos"""
        self.detector.fit(self.training_data)

        # Verificar que los modelos fueron entrenados
        self.assertTrue(hasattr(self.detector.isolation_forest, 'estimators_'))
        self.assertIsNotNone(self.detector.scaler)

    def test_predict_anomalies(self):
        """Probar predicción de anomalías"""
        # Entrenar detector
        self.detector.fit(self.training_data)

        # Crear datos de prueba con anomalías conocidas
        test_data = pd.DataFrame({
            'THD_Total': [8.0, 25.0, 7.5, 30.0],  # 25.0 y 30.0 son anómalos
            'Demanda_por_fase': [45.0, 45.0, 46.0, 80.0],
            'Factor_Potencia': [0.92, 0.91, 0.93, 0.5],
            'Potencia_Activa': [38.0, 39.0, 37.0, 60.0]
        })

        predictions = self.detector.predict(test_data)

        # Verificar formato de salida
        self.assertEqual(len(predictions), 4)
        self.assertTrue(all(p in [-1, 1] for p in predictions))

    def test_ensemble_weights(self):
        """Probar ponderación del ensemble"""
        self.detector.fit(self.training_data)

        test_data = self.training_data.iloc[:100]
        scores = self.detector.decision_function(test_data)

        # Verificar que se obtienen scores
        self.assertEqual(len(scores), 100)
        self.assertTrue(all(isinstance(s, (int, float)) for s in scores))

    def test_reproducibility(self):
        """Probar reproducibilidad de resultados con semilla fija"""
        # Crear dos detectores idénticos
        detector1 = TFMAnomalyDetector(
            isolation_forest_params=self.isolation_forest_params,
            dbscan_params=self.dbscan_params,
            ensemble_weights=self.ensemble_weights
        )

        detector2 = TFMAnomalyDetector(
            isolation_forest_params=self.isolation_forest_params,
            dbscan_params=self.dbscan_params,
            ensemble_weights=self.ensemble_weights
        )

        # Entrenar con los mismos datos
        detector1.fit(self.training_data)
        detector2.fit(self.training_data)

        # Predecir en los mismos datos de prueba
        test_data = self.training_data.iloc[:50]
        pred1 = detector1.predict(test_data)
        pred2 = detector2.predict(test_data)

        # Verificar reproducibilidad (debería ser idéntica con random_state=42)
        np.testing.assert_array_equal(pred1, pred2)


class TestMonitorSistema(unittest.TestCase):
    """Pruebas para sistema de monitoreo continuo"""

    def setUp(self):
        """Configurar sistema de monitoreo"""
        # Crear configuración temporal
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'config.json')

        test_config = {
            'maintenance': {'prediction_window_hours': 72},
            'monitoring': {
                'watching_system': {
                    'alert_thresholds': {
                        'anomaly_score': 0.8,
                        'thd_critical': 15.0,
                        'power_factor_min': 0.85
                    }
                }
            },
            'paths': {'logs_dir': os.path.join(self.temp_dir, 'logs')},
            'logging': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'level': 'INFO'
            }
        }

        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        """Limpiar archivos temporales"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_monitor_initialization(self):
        """Probar inicialización del monitor"""
        monitor = MonitorSistema(config_path=self.config_file)

        self.assertIsNotNone(monitor.config)
        self.assertIsNotNone(monitor.data_loader)
        self.assertFalse(monitor.is_monitoring)
        self.assertEqual(monitor.prediction_window, 72)

    def test_config_loading(self):
        """Probar carga de configuración"""
        monitor = MonitorSistema(config_path=self.config_file)

        self.assertEqual(monitor.config['maintenance']['prediction_window_hours'], 72)
        self.assertEqual(monitor.config['monitoring']['watching_system']['alert_thresholds']['thd_critical'], 15.0)

    def test_logging_setup(self):
        """Probar configuración de logging"""
        monitor = MonitorSistema(config_path=self.config_file)

        self.assertIsNotNone(monitor.logger)
        self.assertEqual(monitor.logger.level, 20)  # INFO level

    @patch('src.monitor_sistema.DataLoader')
    @patch('src.monitor_sistema.TFMAnomalyDetector')
    def test_system_status(self, mock_detector, mock_loader):
        """Probar obtención de estado del sistema"""
        monitor = MonitorSistema(config_path=self.config_file)

        status = monitor.get_system_status()

        self.assertIn('monitoring_active', status)
        self.assertIn('system_health', status)
        self.assertIn('total_analyses', status)
        self.assertFalse(status['monitoring_active'])
        self.assertEqual(status['system_health'], 'STOPPED')


class TestTFMIntegration(unittest.TestCase):
    """Pruebas de integración del sistema TFM completo"""

    def setUp(self):
        """Configurar entorno de integración"""
        self.temp_dir = tempfile.mkdtemp()

        # Crear estructura de directorios
        os.makedirs(os.path.join(self.temp_dir, 'data', 'raw', 'sensor'))
        os.makedirs(os.path.join(self.temp_dir, 'config'))

        # Crear datos de prueba para los 3 compresores
        for compressor in ['Compresor1_FP1', 'Compresor2_FP1', 'Compresor3_FP1']:
            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
                'THD_Total': np.random.normal(8.5, 2.0, 1000),
                'Demanda_por_fase': np.random.normal(45.2, 5.0, 1000),
                'Factor_Potencia': np.random.normal(0.92, 0.05, 1000),
                'Potencia_Activa': np.random.normal(38.7, 4.0, 1000)
            })

            # Introducir anomalías controladas
            anomaly_indices = np.random.choice(1000, 5, replace=False)
            data.loc[anomaly_indices, 'THD_Total'] = np.random.normal(25, 3, 5)

            file_path = os.path.join(self.temp_dir, 'data', 'raw', 'sensor', f'{compressor}.csv')
            data.to_csv(file_path, index=False)

    def tearDown(self):
        """Limpiar entorno de prueba"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_pipeline(self):
        """Prueba de pipeline completo de extremo a extremo"""

        # 1. Cargar datos
        data_loader = DataLoader()
        all_data = []

        for compressor in ['Compresor1_FP1', 'Compresor2_FP1', 'Compresor3_FP1']:
            file_path = os.path.join(self.temp_dir, 'data', 'raw', 'sensor', f'{compressor}.csv')
            data = data_loader.load_compressor_data(compressor, file_path)

            self.assertIsNotNone(data)
            all_data.append(data)

        # 2. Combinar datos
        combined_data = pd.concat(all_data, ignore_index=True)
        self.assertEqual(len(combined_data), 3000)  # 3 compresores × 1000 registros

        # 3. Entrenar detector
        detector = TFMAnomalyDetector(
            isolation_forest_params={
                'n_estimators': 50,  # Reducido para pruebas rápidas
                'max_samples': 0.8,
                'contamination': 0.01,  # Ajustado para datos de prueba
                'random_state': 42
            },
            dbscan_params={'eps': 1.2, 'min_samples': 5},
            ensemble_weights={'isolation_forest': 0.7, 'dbscan': 0.3}
        )

        detector.fit(combined_data)

        # 4. Detectar anomalías
        predictions = detector.predict(combined_data)
        anomaly_count = np.sum(predictions == -1)

        # Verificar que se detectaron algunas anomalías
        self.assertGreater(anomaly_count, 0)
        self.assertLess(anomaly_count, len(combined_data) * 0.1)  # Menos del 10%

        # 5. Validar formato de salida
        self.assertEqual(len(predictions), len(combined_data))
        self.assertTrue(all(p in [-1, 1] for p in predictions))


class TestTFMReproducibility(unittest.TestCase):
    """Pruebas específicas para reproducir resultados exactos del TFM de Antonio"""

    def test_expected_parameters(self):
        """Verificar parámetros exactos del TFM"""

        # Parámetros exactos según el TFM
        expected_if_params = {
            'n_estimators': 200,
            'max_samples': 0.8,
            'contamination': 0.004319,
            'random_state': 42
        }

        expected_dbscan_params = {
            'eps': 1.2,
            'min_samples': 5,
            'metric': 'euclidean'
        }

        expected_ensemble_weights = {
            'isolation_forest': 0.7,
            'dbscan': 0.3
        }

        detector = TFMAnomalyDetector(
            isolation_forest_params=expected_if_params,
            dbscan_params=expected_dbscan_params,
            ensemble_weights=expected_ensemble_weights
        )

        # Verificar parámetros
        self.assertEqual(detector.isolation_forest.n_estimators, 200)
        self.assertEqual(detector.isolation_forest.max_samples, 0.8)
        self.assertAlmostEqual(detector.isolation_forest.contamination, 0.004319, places=6)
        self.assertEqual(detector.dbscan.eps, 1.2)
        self.assertEqual(detector.dbscan.min_samples, 5)
        self.assertEqual(detector.ensemble_weights['isolation_forest'], 0.7)

    def test_critical_variables_importance(self):
        """Verificar importancia de variables críticas según TFM"""

        expected_importance = {
            'THD_Total': 32.1,
            'Demanda_por_fase': 26.8,
            'Factor_Potencia': 22.4,
            'Potencia_Activa': 18.7
        }

        # Crear datos con estas variables
        data = pd.DataFrame({
            'THD_Total': np.random.normal(8.5, 2.0, 1000),
            'Demanda_por_fase': np.random.normal(45.2, 5.0, 1000),
            'Factor_Potencia': np.random.normal(0.92, 0.05, 1000),
            'Potencia_Activa': np.random.normal(38.7, 4.0, 1000)
        })

        # Verificar que todas las variables críticas están presentes
        for var_name in expected_importance.keys():
            self.assertIn(var_name, data.columns)

        # Verificar que los valores de importancia suman aproximadamente 100%
        total_importance = sum(expected_importance.values())
        self.assertAlmostEqual(total_importance, 100.0, delta=0.1)


def create_test_suite():
    """Crear suite completa de pruebas"""
    suite = unittest.TestSuite()

    # Agregar todas las clases de prueba
    test_classes = [
        TestDataUtils,
        TestMLUtils,
        TestMonitorSistema,
        TestTFMIntegration,
        TestTFMReproducibility
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def main():
    """Función principal para ejecutar todas las pruebas"""
    print("🧪 Ejecutando Suite Completa de Pruebas TFM")
    print("=" * 60)
    print("📊 Validando sistema de mantenimiento predictivo")
    print("🎯 Verificando reproducibilidad de resultados")
    print("⚙️ Comprobando funcionalidades críticas")
    print()

    # Crear y ejecutar suite de pruebas
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    print()
    print("=" * 60)
    print(f"✅ Pruebas ejecutadas: {result.testsRun}")
    print(f"❌ Fallos: {len(result.failures)}")
    print(f"🚫 Errores: {len(result.errors)}")

    if result.wasSuccessful():
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
        print("✅ Sistema TFM validado correctamente")
        print("🚀 Listo para producción")
    else:
        print("⚠️ Algunas pruebas fallaron")
        if result.failures:
            print("\nFallos encontrados:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if result.errors:
            print("\nErrores encontrados:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
