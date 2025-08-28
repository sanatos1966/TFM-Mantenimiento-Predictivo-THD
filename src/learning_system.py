
"""
Sistema de Aprendizaje Continuo para Mantenimiento Predictivo
=============================================================

Implementa capacidades de aprendizaje automático y adaptación del modelo
basado en nuevos datos industriales de compresores.

Autor: Antonio (TFM)
Fecha: Enero 2025
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import logging

class SistemaAprendizajeContinuo:
    """
    Sistema de aprendizaje continuo que adapta los modelos de mantenimiento
    predictivo basándose en nuevos datos y retroalimentación del sistema.
    """

    def __init__(self, config_path="config/config.json"):
        """
        Inicializa el sistema de aprendizaje continuo.

        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.cargar_configuracion()
        self.configurar_logging()
        self.modelos_entrenados = {}
        self.historial_rendimiento = []
        self.umbral_reentrenamiento = 0.05  # 5% degradación dispara reentrenamiento

    def cargar_configuracion(self):
        """Carga la configuración desde el archivo JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.logger = logging.getLogger(__name__)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo de configuración {self.config_path}")
            raise

    def configurar_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def cargar_modelos_existentes(self, ruta_modelos="models/"):
        """
        Carga modelos previamente entrenados desde disco.

        Args:
            ruta_modelos (str): Ruta donde están almacenados los modelos
        """
        try:
            if os.path.exists(os.path.join(ruta_modelos, "isolation_forest.pkl")):
                self.modelos_entrenados['isolation_forest'] = joblib.load(
                    os.path.join(ruta_modelos, "isolation_forest.pkl")
                )
                self.logger.info("Modelo Isolation Forest cargado exitosamente")

            if os.path.exists(os.path.join(ruta_modelos, "dbscan.pkl")):
                self.modelos_entrenados['dbscan'] = joblib.load(
                    os.path.join(ruta_modelos, "dbscan.pkl")
                )
                self.logger.info("Modelo DBSCAN cargado exitosamente")

            if os.path.exists(os.path.join(ruta_modelos, "historial_rendimiento.pkl")):
                self.historial_rendimiento = joblib.load(
                    os.path.join(ruta_modelos, "historial_rendimiento.pkl")
                )
                self.logger.info("Historial de rendimiento cargado exitosamente")

        except Exception as e:
            self.logger.error(f"Error cargando modelos: {str(e)}")

    def evaluar_rendimiento_actual(self, datos_nuevos, etiquetas_verdaderas=None):
        """
        Evalúa el rendimiento actual de los modelos con nuevos datos.

        Args:
            datos_nuevos (pd.DataFrame): Nuevos datos para evaluación
            etiquetas_verdaderas (np.array): Etiquetas verdaderas si están disponibles

        Returns:
            dict: Métricas de rendimiento
        """
        metricas = {}

        try:
            # Preparar datos
            datos_procesados = self.preparar_datos_para_modelo(datos_nuevos)

            # Evaluar Isolation Forest
            if 'isolation_forest' in self.modelos_entrenados:
                predicciones_if = self.modelos_entrenados['isolation_forest'].predict(datos_procesados)
                predicciones_if = np.where(predicciones_if == -1, 1, 0)  # Convertir a anomalías

                if etiquetas_verdaderas is not None:
                    metricas['isolation_forest'] = {
                        'f1_score': f1_score(etiquetas_verdaderas, predicciones_if),
                        'precision': precision_score(etiquetas_verdaderas, predicciones_if),
                        'recall': recall_score(etiquetas_verdaderas, predicciones_if)
                    }

            # Evaluar DBSCAN (clustering)
            if 'dbscan' in self.modelos_entrenados:
                predicciones_dbscan = self.modelos_entrenados['dbscan'].fit_predict(datos_procesados)
                predicciones_dbscan = np.where(predicciones_dbscan == -1, 1, 0)  # Outliers como anomalías

                if etiquetas_verdaderas is not None:
                    metricas['dbscan'] = {
                        'f1_score': f1_score(etiquetas_verdaderas, predicciones_dbscan),
                        'precision': precision_score(etiquetas_verdaderas, predicciones_dbscan),
                        'recall': recall_score(etiquetas_verdaderas, predicciones_dbscan)
                    }

            # Evaluación del ensemble
            if 'isolation_forest' in self.modelos_entrenados and 'dbscan' in self.modelos_entrenados:
                ensemble_pred = self.combinar_predicciones(predicciones_if, predicciones_dbscan)

                if etiquetas_verdaderas is not None:
                    metricas['ensemble'] = {
                        'f1_score': f1_score(etiquetas_verdaderas, ensemble_pred),
                        'precision': precision_score(etiquetas_verdaderas, ensemble_pred),
                        'recall': recall_score(etiquetas_verdaderas, ensemble_pred)
                    }

            # Registrar métricas en historial
            entrada_historial = {
                'timestamp': datetime.now().isoformat(),
                'metricas': metricas,
                'num_muestras': len(datos_nuevos)
            }
            self.historial_rendimiento.append(entrada_historial)

            self.logger.info(f"Evaluación de rendimiento completada: {len(datos_nuevos)} muestras")

        except Exception as e:
            self.logger.error(f"Error en evaluación de rendimiento: {str(e)}")

        return metricas

    def detectar_degradacion_modelo(self):
        """
        Detecta si el rendimiento del modelo ha degradado significativamente.

        Returns:
            bool: True si se detecta degradación significativa
        """
        if len(self.historial_rendimiento) < 2:
            return False

        try:
            # Comparar últimas dos evaluaciones
            rendimiento_actual = self.historial_rendimiento[-1]['metricas']
            rendimiento_anterior = self.historial_rendimiento[-2]['metricas']

            # Verificar degradación en F1-score del ensemble
            if 'ensemble' in rendimiento_actual and 'ensemble' in rendimiento_anterior:
                f1_actual = rendimiento_actual['ensemble']['f1_score']
                f1_anterior = rendimiento_anterior['ensemble']['f1_score']

                degradacion = f1_anterior - f1_actual

                if degradacion > self.umbral_reentrenamiento:
                    self.logger.warning(f"Degradación detectada: {degradacion:.3f} > {self.umbral_reentrenamiento}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error detectando degradación: {str(e)}")
            return False

    def reentrenar_modelos(self, datos_historicos, datos_nuevos):
        """
        Reentrenar modelos combinando datos históricos con nuevos datos.

        Args:
            datos_historicos (pd.DataFrame): Datos históricos de entrenamiento
            datos_nuevos (pd.DataFrame): Nuevos datos para incorporar

        Returns:
            dict: Información sobre el reentrenamiento
        """
        info_reentrenamiento = {
            'timestamp': datetime.now().isoformat(),
            'datos_historicos': len(datos_historicos),
            'datos_nuevos': len(datos_nuevos),
            'exitoso': False
        }

        try:
            # Combinar datasets
            datos_combinados = pd.concat([datos_historicos, datos_nuevos], ignore_index=True)
            datos_procesados = self.preparar_datos_para_modelo(datos_combinados)

            # Reentrenar Isolation Forest
            modelo_if = IsolationForest(
                n_estimators=self.config['modelos']['isolation_forest']['n_estimators'],
                max_samples=self.config['modelos']['isolation_forest']['max_samples'],
                contamination=self.config['modelos']['isolation_forest']['contamination'],
                random_state=42,
                n_jobs=-1
            )

            modelo_if.fit(datos_procesados)
            self.modelos_entrenados['isolation_forest'] = modelo_if

            # Reentrenar DBSCAN (ajustar parámetros si es necesario)
            modelo_dbscan = DBSCAN(
                eps=self.config['modelos']['dbscan']['eps'],
                min_samples=self.config['modelos']['dbscan']['min_samples'],
                n_jobs=-1
            )

            self.modelos_entrenados['dbscan'] = modelo_dbscan

            # Guardar modelos actualizados
            self.guardar_modelos()

            info_reentrenamiento['exitoso'] = True
            self.logger.info(f"Reentrenamiento completado exitosamente con {len(datos_combinados)} muestras")

        except Exception as e:
            self.logger.error(f"Error en reentrenamiento: {str(e)}")
            info_reentrenamiento['error'] = str(e)

        return info_reentrenamiento

    def aprendizaje_incremental(self, datos_nuevos, ventana_temporal=30):
        """
        Implementa aprendizaje incremental con ventana temporal deslizante.

        Args:
            datos_nuevos (pd.DataFrame): Nuevos datos para aprender
            ventana_temporal (int): Días de ventana temporal para el aprendizaje

        Returns:
            dict: Información sobre el proceso de aprendizaje
        """
        info_aprendizaje = {
            'timestamp': datetime.now().isoformat(),
            'muestras_procesadas': len(datos_nuevos),
            'aprendizaje_aplicado': False
        }

        try:
            # Filtrar datos por ventana temporal
            fecha_limite = datetime.now() - timedelta(days=ventana_temporal)

            if 'timestamp' in datos_nuevos.columns:
                datos_nuevos['timestamp'] = pd.to_datetime(datos_nuevos['timestamp'])
                datos_filtrados = datos_nuevos[datos_nuevos['timestamp'] >= fecha_limite]
            else:
                datos_filtrados = datos_nuevos.tail(int(len(datos_nuevos) * 0.3))  # Último 30%

            # Actualizar parámetros del modelo basándose en características de nuevos datos
            caracteristicas_nuevas = self.analizar_caracteristicas_datos(datos_filtrados)

            # Ajustar parámetros si es necesario
            ajustes_realizados = self.ajustar_parametros_modelo(caracteristicas_nuevas)

            if ajustes_realizados:
                info_aprendizaje['aprendizaje_aplicado'] = True
                info_aprendizaje['ajustes'] = ajustes_realizados

                # Guardar configuración actualizada
                self.guardar_configuracion_actualizada()

            self.logger.info(f"Aprendizaje incremental procesado: {len(datos_filtrados)} muestras")

        except Exception as e:
            self.logger.error(f"Error en aprendizaje incremental: {str(e)}")
            info_aprendizaje['error'] = str(e)

        return info_aprendizaje

    def analizar_caracteristicas_datos(self, datos):
        """
        Analiza las características estadísticas de los nuevos datos.

        Args:
            datos (pd.DataFrame): Datos para analizar

        Returns:
            dict: Características estadísticas
        """
        caracteristicas = {}

        try:
            # Variables numéricas clave del TFM
            variables_clave = ['THD_I_L1(%)', 'THD_V_L1(%)', 'Factor_Potencia', 
                             'Corriente_L1(A)', 'Vibracion_Axial']

            for var in variables_clave:
                if var in datos.columns:
                    caracteristicas[var] = {
                        'media': datos[var].mean(),
                        'std': datos[var].std(),
                        'percentil_95': datos[var].quantile(0.95),
                        'outliers_ratio': len(datos[datos[var] > datos[var].quantile(0.99)]) / len(datos)
                    }

            # Análisis de correlaciones
            if len(variables_clave) >= 2:
                vars_disponibles = [v for v in variables_clave if v in datos.columns]
                if len(vars_disponibles) >= 2:
                    correlaciones = datos[vars_disponibles].corr()
                    caracteristicas['correlaciones_promedio'] = correlaciones.abs().mean().mean()

        except Exception as e:
            self.logger.error(f"Error analizando características: {str(e)}")

        return caracteristicas

    def ajustar_parametros_modelo(self, caracteristicas):
        """
        Ajusta parámetros del modelo basándose en características de los datos.

        Args:
            caracteristicas (dict): Características estadísticas de los datos

        Returns:
            dict: Ajustes realizados
        """
        ajustes = {}

        try:
            # Ajustar contaminación de Isolation Forest basándose en outliers
            if 'THD_I_L1(%)' in caracteristicas:
                outliers_ratio = caracteristicas['THD_I_L1(%)']['outliers_ratio']

                if outliers_ratio > 0.1:  # Si hay muchos outliers
                    nueva_contaminacion = min(0.01, outliers_ratio * 0.5)
                    if abs(nueva_contaminacion - self.config['modelos']['isolation_forest']['contamination']) > 0.001:
                        self.config['modelos']['isolation_forest']['contamination'] = nueva_contaminacion
                        ajustes['isolation_forest_contamination'] = nueva_contaminacion

            # Ajustar eps de DBSCAN basándose en dispersión de datos
            if 'correlaciones_promedio' in caracteristicas:
                corr_promedio = caracteristicas['correlaciones_promedio']

                if corr_promedio < 0.3:  # Datos muy dispersos
                    nuevo_eps = self.config['modelos']['dbscan']['eps'] * 1.1
                    self.config['modelos']['dbscan']['eps'] = min(nuevo_eps, 2.0)
                    ajustes['dbscan_eps'] = nuevo_eps
                elif corr_promedio > 0.7:  # Datos muy correlacionados
                    nuevo_eps = self.config['modelos']['dbscan']['eps'] * 0.9
                    self.config['modelos']['dbscan']['eps'] = max(nuevo_eps, 0.5)
                    ajustes['dbscan_eps'] = nuevo_eps

        except Exception as e:
            self.logger.error(f"Error ajustando parámetros: {str(e)}")

        return ajustes

    def preparar_datos_para_modelo(self, datos):
        """
        Prepara los datos para el modelo (normalización, selección de features).

        Args:
            datos (pd.DataFrame): Datos a preparar

        Returns:
            np.array: Datos preparados
        """
        try:
            # Seleccionar variables clave del TFM
            variables_modelo = ['THD_I_L1(%)', 'THD_V_L1(%)', 'Factor_Potencia', 
                              'Corriente_L1(A)', 'Vibracion_Axial']

            # Filtrar variables disponibles
            variables_disponibles = [var for var in variables_modelo if var in datos.columns]

            if not variables_disponibles:
                raise ValueError("No se encontraron variables necesarias para el modelo")

            datos_modelo = datos[variables_disponibles].copy()

            # Limpiar datos
            datos_modelo = datos_modelo.dropna()

            # Normalización simple (Z-score)
            for col in datos_modelo.columns:
                if datos_modelo[col].std() > 0:
                    datos_modelo[col] = (datos_modelo[col] - datos_modelo[col].mean()) / datos_modelo[col].std()

            return datos_modelo.values

        except Exception as e:
            self.logger.error(f"Error preparando datos: {str(e)}")
            raise

    def combinar_predicciones(self, pred_if, pred_dbscan, peso_if=0.7, peso_dbscan=0.3):
        """
        Combina predicciones de Isolation Forest y DBSCAN usando pesos del TFM.

        Args:
            pred_if (np.array): Predicciones de Isolation Forest
            pred_dbscan (np.array): Predicciones de DBSCAN
            peso_if (float): Peso para Isolation Forest (0.7 según TFM)
            peso_dbscan (float): Peso para DBSCAN (0.3 según TFM)

        Returns:
            np.array: Predicciones combinadas
        """
        try:
            # Combinar con pesos especificados en el TFM
            puntuaciones_combinadas = (peso_if * pred_if) + (peso_dbscan * pred_dbscan)
            predicciones_finales = np.where(puntuaciones_combinadas > 0.5, 1, 0)

            return predicciones_finales

        except Exception as e:
            self.logger.error(f"Error combinando predicciones: {str(e)}")
            return pred_if  # Fallback a Isolation Forest

    def guardar_modelos(self, ruta_modelos="models/"):
        """
        Guarda los modelos entrenados en disco.

        Args:
            ruta_modelos (str): Ruta donde guardar los modelos
        """
        try:
            os.makedirs(ruta_modelos, exist_ok=True)

            if 'isolation_forest' in self.modelos_entrenados:
                joblib.dump(
                    self.modelos_entrenados['isolation_forest'],
                    os.path.join(ruta_modelos, "isolation_forest.pkl")
                )

            if 'dbscan' in self.modelos_entrenados:
                joblib.dump(
                    self.modelos_entrenados['dbscan'],
                    os.path.join(ruta_modelos, "dbscan.pkl")
                )

            # Guardar historial de rendimiento
            joblib.dump(
                self.historial_rendimiento,
                os.path.join(ruta_modelos, "historial_rendimiento.pkl")
            )

            self.logger.info("Modelos guardados exitosamente")

        except Exception as e:
            self.logger.error(f"Error guardando modelos: {str(e)}")

    def guardar_configuracion_actualizada(self):
        """Guarda la configuración actualizada en disco."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            self.logger.info("Configuración actualizada guardada")

        except Exception as e:
            self.logger.error(f"Error guardando configuración: {str(e)}")

    def obtener_reporte_aprendizaje(self):
        """
        Genera un reporte del estado del sistema de aprendizaje.

        Returns:
            dict: Reporte completo del sistema
        """
        reporte = {
            'timestamp': datetime.now().isoformat(),
            'modelos_cargados': list(self.modelos_entrenados.keys()),
            'evaluaciones_realizadas': len(self.historial_rendimiento),
            'rendimiento_actual': None,
            'degradacion_detectada': False,
            'ultima_actualizacion': None
        }

        try:
            if self.historial_rendimiento:
                reporte['rendimiento_actual'] = self.historial_rendimiento[-1]['metricas']
                reporte['ultima_actualizacion'] = self.historial_rendimiento[-1]['timestamp']
                reporte['degradacion_detectada'] = self.detectar_degradacion_modelo()

            # Estadísticas del historial
            if len(self.historial_rendimiento) > 1:
                f1_scores = []
                for evaluacion in self.historial_rendimiento:
                    if 'ensemble' in evaluacion['metricas']:
                        f1_scores.append(evaluacion['metricas']['ensemble']['f1_score'])

                if f1_scores:
                    reporte['estadisticas_f1'] = {
                        'promedio': np.mean(f1_scores),
                        'minimo': np.min(f1_scores),
                        'maximo': np.max(f1_scores),
                        'tendencia': 'mejorando' if f1_scores[-1] > f1_scores[0] else 'degradando'
                    }

        except Exception as e:
            self.logger.error(f"Error generando reporte: {str(e)}")
            reporte['error'] = str(e)

        return reporte


# Función auxiliar para integración con el sistema principal
def inicializar_sistema_aprendizaje(config_path="config/config.json"):
    """
    Función auxiliar para inicializar el sistema de aprendizaje.

    Args:
        config_path (str): Ruta al archivo de configuración

    Returns:
        SistemaAprendizajeContinuo: Instancia del sistema
    """
    sistema = SistemaAprendizajeContinuo(config_path)
    sistema.cargar_modelos_existentes()
    return sistema


if __name__ == "__main__":
    # Ejemplo de uso del sistema de aprendizaje
    print("Sistema de Aprendizaje Continuo para Mantenimiento Predictivo")
    print("=" * 60)

    # Inicializar sistema
    sistema = inicializar_sistema_aprendizaje()

    # Generar reporte
    reporte = sistema.obtener_reporte_aprendizaje()
    print(f"Reporte del sistema: {json.dumps(reporte, indent=2, ensure_ascii=False)}")
