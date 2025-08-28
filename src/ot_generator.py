#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
GENERADOR AUTOMÁTICO DE ÓRDENES DE TRABAJO - MANTENIMIENTO INTELIGENTE
===============================================================================
Autor: Antonio Cantos & Renzo Chavez - TFM EADIC
Descripción: Sistema que analiza anomalías y genera OT automáticamente
Tipos: Correctivo, Preventivo, Predictivo, Prescriptivo
===============================================================================
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

class GeneradorOrdenesTrabajo:
    """Sistema inteligente de generación automática de órdenes de trabajo"""

    def __init__(self, config_path="config/config.json"):
        """Inicializar generador de OT"""
        self.cargar_configuracion(config_path)
        self.configurar_logging()

        # Base de conocimientos para diagnóstico
        self.base_conocimientos = self.crear_base_conocimientos()
        self.ot_generadas = []

        self.logger.info("📋 Generador de Órdenes de Trabajo iniciado")

    def cargar_configuracion(self, config_path):
        """Cargar configuración del sistema"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {"mantenimiento": {"tipos_ot": ["Correctivo", "Preventivo", "Predictivo", "Prescriptivo"]}}

    def configurar_logging(self):
        """Configurar sistema de logging"""
        self.logger = logging.getLogger('OT_Generator')
        if not self.logger.handlers:
            handler = logging.FileHandler('logs/ot_generator.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def crear_base_conocimientos(self):
        """Crear base de conocimientos para diagnóstico inteligente"""
        return {
            'THD_Alto': {
                'umbral': 5.0,
                'tipo_mantenimiento': 'Predictivo',
                'prioridad': 'Alta',
                'descripcion': 'THD elevado detectado - Posible desalineación o problema VDF',
                'acciones': [
                    'Inspeccionar conexiones eléctricas',
                    'Verificar alineación del motor',
                    'Revisar estado del variador de frecuencia',
                    'Medir resistencia de aislamiento'
                ],
                'tiempo_estimado': 4,
                'especialista': 'Eléctrico',
                'repuestos': ['Terminales eléctricos', 'Aisladores']
            },
            'Factor_Potencia_Bajo': {
                'umbral': 0.7,
                'tipo_mantenimiento': 'Correctivo',
                'prioridad': 'Crítica',
                'descripcion': 'Factor de potencia bajo - Problema en capacitores o motor',
                'acciones': [
                    'Revisar estado de capacitores',
                    'Verificar conexiones del motor',
                    'Medir corrientes por fase',
                    'Inspeccionar rotor y estator'
                ],
                'tiempo_estimado': 6,
                'especialista': 'Eléctrico',
                'repuestos': ['Capacitores', 'Contactores']
            },
            'Vibracion_Elevada': {
                'umbral': 7.0,
                'tipo_mantenimiento': 'Correctivo',
                'prioridad': 'Alta',
                'descripcion': 'Vibración excesiva - Posible problema mecánico',
                'acciones': [
                    'Verificar anclajes y soportes',
                    'Inspeccionar rodamientos',
                    'Revisar alineación de ejes',
                    'Análisis de lubricación'
                ],
                'tiempo_estimado': 8,
                'especialista': 'Mecánico',
                'repuestos': ['Rodamientos', 'Lubricante', 'Retenes']
            },
            'Potencia_Anomala': {
                'umbral_min': 30,
                'umbral_max': 70,
                'tipo_mantenimiento': 'Predictivo',
                'prioridad': 'Media',
                'descripcion': 'Consumo de potencia anómalo - Eficiencia comprometida',
                'acciones': [
                    'Verificar presiones de operación',
                    'Revisar estado de válvulas',
                    'Inspeccionar intercambiadores',
                    'Análisis de eficiencia energética'
                ],
                'tiempo_estimado': 3,
                'especialista': 'Frigorista',
                'repuestos': ['Filtros', 'Válvulas', 'Refrigerante']
            },
            'Correlacion_THD_Vibracion': {
                'correlacion_min': 0.6,
                'tipo_mantenimiento': 'Prescriptivo',
                'prioridad': 'Alta',
                'descripcion': 'Correlación THD-Vibración detectada - Problema sistémico',
                'acciones': [
                    'Análisis integral eléctrico-mecánico',
                    'Verificar acoplamiento motor-compresor',
                    'Revisar base y cimentación',
                    'Balanceo dinámico si necesario'
                ],
                'tiempo_estimado': 12,
                'especialista': 'Multidisciplinario',
                'repuestos': ['Acoples flexibles', 'Elementos antivibratorios']
            },
            'Mantenimiento_Preventivo': {
                'frecuencia_dias': 90,
                'tipo_mantenimiento': 'Preventivo',
                'prioridad': 'Media',
                'descripcion': 'Mantenimiento preventivo programado',
                'acciones': [
                    'Limpieza general del equipo',
                    'Verificación de parámetros operativos',
                    'Lubricación según programa',
                    'Inspección visual general'
                ],
                'tiempo_estimado': 2,
                'especialista': 'Operario',
                'repuestos': ['Materiales de limpieza', 'Lubricantes']
            }
        }

    def analizar_anomalias(self, df_datos, anomalias_detectadas):
        """Analiza las anomalías detectadas y genera diagnósticos"""
        self.logger.info("🔍 Analizando anomalías para generación de OT...")

        diagnosticos = []
        indices_anomalias = np.where(anomalias_detectadas)[0]

        for idx in indices_anomalias:
            registro = df_datos.iloc[idx]
            diagnostico = self.diagnosticar_registro(registro)
            if diagnostico:
                diagnosticos.append(diagnostico)

        self.logger.info(f"✅ {len(diagnosticos)} diagnósticos generados")
        return diagnosticos

    def diagnosticar_registro(self, registro):
        """Diagnostica un registro específico y determina el tipo de mantenimiento"""
        diagnosticos_encontrados = []

        # Verificar THD elevado
        if registro['THD_Total'] > self.base_conocimientos['THD_Alto']['umbral']:
            diagnosticos_encontrados.append(('THD_Alto', registro['THD_Total']))

        # Verificar factor de potencia bajo
        if registro['Factor_Potencia'] < self.base_conocimientos['Factor_Potencia_Bajo']['umbral']:
            diagnosticos_encontrados.append(('Factor_Potencia_Bajo', registro['Factor_Potencia']))

        # Verificar vibración elevada
        if registro['Vibracion_RMS_Vectorial'] > self.base_conocimientos['Vibracion_Elevada']['umbral']:
            diagnosticos_encontrados.append(('Vibracion_Elevada', registro['Vibracion_RMS_Vectorial']))

        # Verificar potencia anómala
        umbral_pot_min = self.base_conocimientos['Potencia_Anomala']['umbral_min']
        umbral_pot_max = self.base_conocimientos['Potencia_Anomala']['umbral_max']
        if not (umbral_pot_min <= registro['Potencia_Activa'] <= umbral_pot_max):
            diagnosticos_encontrados.append(('Potencia_Anomala', registro['Potencia_Activa']))

        # Si hay múltiples problemas, considerar diagnóstico sistémico
        if len(diagnosticos_encontrados) >= 2:
            # Verificar correlación THD-Vibración
            if any('THD' in d[0] for d in diagnosticos_encontrados) and any('Vibracion' in d[0] for d in diagnosticos_encontrados):
                diagnosticos_encontrados.append(('Correlacion_THD_Vibracion', len(diagnosticos_encontrados)))

        if diagnosticos_encontrados:
            # Seleccionar el diagnóstico más crítico
            diagnostico_principal = self.seleccionar_diagnostico_principal(diagnosticos_encontrados)
            return {
                'compresor_id': registro['Compresor_ID'],
                'diagnostico': diagnostico_principal,
                'valores_anomalos': diagnosticos_encontrados,
                'timestamp': registro.get('Timestamp', datetime.now()),
                'indice_registro': registro.name if hasattr(registro, 'name') else None
            }

        return None

    def seleccionar_diagnostico_principal(self, diagnosticos):
        """Selecciona el diagnóstico principal basado en prioridad"""
        prioridades = {'Crítica': 4, 'Alta': 3, 'Media': 2, 'Baja': 1}

        diagnostico_seleccionado = None
        max_prioridad = 0

        for diagnostico, valor in diagnosticos:
            conocimiento = self.base_conocimientos.get(diagnostico, {})
            prioridad_texto = conocimiento.get('prioridad', 'Baja')
            prioridad_numerica = prioridades.get(prioridad_texto, 1)

            if prioridad_numerica > max_prioridad:
                max_prioridad = prioridad_numerica
                diagnostico_seleccionado = diagnostico

        return diagnostico_seleccionado

    def generar_ot_inteligente(self, diagnostico):
        """Genera una orden de trabajo inteligente basada en el diagnóstico"""
        conocimiento = self.base_conocimientos[diagnostico['diagnostico']]

        # Generar ID único de OT
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ot_id = f"FP1-{diagnostico['compresor_id']}-{timestamp}"

        # Calcular fecha objetivo basada en prioridad
        fecha_objetivo = self.calcular_fecha_objetivo(conocimiento['prioridad'])

        # Generar explicación detallada
        explicacion = self.generar_explicacion_detallada(diagnostico, conocimiento)

        ot = {
            'ot_id': ot_id,
            'fecha_creacion': datetime.now().isoformat(),
            'fecha_objetivo': fecha_objetivo.isoformat(),
            'compresor_id': diagnostico['compresor_id'],
            'tipo_mantenimiento': conocimiento['tipo_mantenimiento'],
            'prioridad': conocimiento['prioridad'],
            'estado': 'Pendiente',
            'titulo': conocimiento['descripcion'],
            'diagnostico_principal': diagnostico['diagnostico'],
            'valores_anomalos': diagnostico['valores_anomalos'],
            'explicacion_detallada': explicacion,
            'acciones_recomendadas': conocimiento['acciones'],
            'tiempo_estimado_horas': conocimiento['tiempo_estimado'],
            'especialista_requerido': conocimiento['especialista'],
            'repuestos_sugeridos': conocimiento['repuestos'],
            'metodo_deteccion': 'IA_Ensemble_TFM',
            'anticipacion_horas': self.calcular_anticipacion(diagnostico),
            'confianza_diagnostico': self.calcular_confianza(diagnostico),
            'recomendaciones_adicionales': self.generar_recomendaciones_adicionales(diagnostico)
        }

        return ot

    def calcular_fecha_objetivo(self, prioridad):
        """Calcula fecha objetivo basada en prioridad"""
        horas_objetivo = {
            'Crítica': 4,    # 4 horas
            'Alta': 24,      # 1 día
            'Media': 72,     # 3 días
            'Baja': 168      # 1 semana
        }

        horas = horas_objetivo.get(prioridad, 72)
        return datetime.now() + timedelta(hours=horas)

    def calcular_anticipacion(self, diagnostico):
        """Calcula horas de anticipación del diagnóstico"""
        # Basado en el análisis del TFM (72h anticipación)
        if 'THD' in diagnostico['diagnostico']:
            return 72  # THD permite mayor anticipación
        elif 'Vibracion' in diagnostico['diagnostico']:
            return 48  # Vibraciones dan menos tiempo
        else:
            return 24  # Otros problemas

    def calcular_confianza(self, diagnostico):
        """Calcula nivel de confianza del diagnóstico"""
        num_anomalias = len(diagnostico['valores_anomalos'])

        if num_anomalias >= 3:
            return 0.95  # Alta confianza
        elif num_anomalias == 2:
            return 0.85  # Confianza media-alta
        else:
            return 0.75  # Confianza media

    def generar_explicacion_detallada(self, diagnostico, conocimiento):
        """Genera explicación detallada del diagnóstico"""
        explicacion = f"""
DIAGNÓSTICO INTELIGENTE - {conocimiento['descripcion']}

ANÁLISIS DE LA ANOMALÍA:
El sistema de IA ha detectado patrones anómalos en {diagnostico['compresor_id']} que requieren atención.

VALORES DETECTADOS:
{self.formatear_valores_anomalos(diagnostico['valores_anomalos'])}

INTERPRETACIÓN TÉCNICA:
{self.generar_interpretacion_tecnica(diagnostico)}

IMPACTO POTENCIAL:
- Riesgo de falla: {self.evaluar_riesgo_falla(diagnostico)}
- Efecto en operación: {self.evaluar_impacto_operacion(diagnostico)}
- Criticidad: {conocimiento['prioridad']}

METODOLOGÍA DE DETECCIÓN:
El algoritmo ensemble (Isolation Forest + DBSCAN) identificó este patrón basándose en:
- Correlaciones entre variables eléctricas y mecánicas
- Patrones temporales de degradación
- Comparación con base histórica de 101,646 registros

VENTANA DE OPORTUNIDAD:
Se dispone de aproximadamente {self.calcular_anticipacion(diagnostico)} horas para realizar la intervención antes de que la condición se agrave.
        """.strip()

        return explicacion

    def formatear_valores_anomalos(self, valores_anomalos):
        """Formatea los valores anómalos para mostrar en la explicación"""
        lineas = []
        for diagnostico, valor in valores_anomalos:
            if 'THD' in diagnostico:
                lineas.append(f"- THD Total: {valor:.2f}% (Normal: <5%)")
            elif 'Factor_Potencia' in diagnostico:
                lineas.append(f"- Factor de Potencia: {valor:.2f} (Normal: >0.8)")
            elif 'Vibracion' in diagnostico:
                lineas.append(f"- Vibración RMS: {valor:.2f} mm/s (Normal: <7 mm/s)")
            elif 'Potencia' in diagnostico:
                lineas.append(f"- Potencia Activa: {valor:.1f} kW (Rango normal: 30-70 kW)")

        return "\n".join(lineas)

    def generar_interpretacion_tecnica(self, diagnostico):
        """Genera interpretación técnica específica"""
        interpretaciones = {
            'THD_Alto': "El incremento en distorsión armónica indica posibles problemas en el sistema eléctrico que pueden preceder fallos mecánicos hasta 72 horas antes.",
            'Factor_Potencia_Bajo': "La reducción del factor de potencia sugiere problemas en capacitores o devanados que requieren atención inmediata.",
            'Vibracion_Elevada': "Las vibraciones excesivas indican problemas mecánicos que pueden causar daños secundarios si no se atienden.",
            'Potencia_Anomala': "El consumo anómalo de potencia sugiere pérdida de eficiencia que puede indicar problemas en el sistema de refrigeración.",
            'Correlacion_THD_Vibracion': "La correlación entre problemas eléctricos y mecánicos indica un problema sistémico que requiere análisis integral."
        }

        return interpretaciones.get(diagnostico['diagnostico'], "Se requiere análisis adicional para determinar la causa raíz.")

    def evaluar_riesgo_falla(self, diagnostico):
        """Evalúa el riesgo de falla basado en el diagnóstico"""
        riesgos = {
            'THD_Alto': 'Medio-Alto',
            'Factor_Potencia_Bajo': 'Alto',
            'Vibracion_Elevada': 'Alto',
            'Potencia_Anomala': 'Medio',
            'Correlacion_THD_Vibracion': 'Muy Alto'
        }
        return riesgos.get(diagnostico['diagnostico'], 'Medio')

    def evaluar_impacto_operacion(self, diagnostico):
        """Evalúa el impacto en la operación"""
        impactos = {
            'THD_Alto': 'Reducción de eficiencia energética',
            'Factor_Potencia_Bajo': 'Posible disparo por protecciones eléctricas',
            'Vibracion_Elevada': 'Daños en equipos adyacentes',
            'Potencia_Anomala': 'Pérdida de capacidad de refrigeración',
            'Correlacion_THD_Vibracion': 'Riesgo de parada total del compresor'
        }
        return impactos.get(diagnostico['diagnostico'], 'Reducción de confiabilidad')

    def generar_recomendaciones_adicionales(self, diagnostico):
        """Genera recomendaciones adicionales específicas"""
        recomendaciones = []

        if 'THD' in diagnostico['diagnostico']:
            recomendaciones.extend([
                "Verificar calidad de energía en acometida",
                "Considerar instalación de filtros armónicos",
                "Revisar conexiones de neutro y tierra"
            ])

        if 'Vibracion' in diagnostico['diagnostico']:
            recomendaciones.extend([
                "Análisis de aceite para detección de partículas metálicas",
                "Termografía de rodamientos",
                "Verificar torque de pernos de anclaje"
            ])

        if len(diagnostico['valores_anomalos']) > 1:
            recomendaciones.append("Considerar análisis de vibración avanzado con FFT")

        return recomendaciones

    def procesar_anomalias_y_generar_ot(self, df_datos, anomalias_detectadas):
        """Proceso completo: analizar anomalías y generar OT"""
        self.logger.info("🚀 Iniciando generación automática de OT...")

        # Analizar anomalías
        diagnosticos = self.analizar_anomalias(df_datos, anomalias_detectadas)

        # Generar OT para cada diagnóstico
        ot_generadas = []
        for diagnostico in diagnosticos:
            ot = self.generar_ot_inteligente(diagnostico)
            ot_generadas.append(ot)

        # Generar OT preventivas programadas
        ot_preventivas = self.generar_ot_preventivas(df_datos)
        ot_generadas.extend(ot_preventivas)

        self.ot_generadas = ot_generadas

        # Guardar OT
        self.guardar_ot_generadas()

        self.logger.info(f"✅ {len(ot_generadas)} órdenes de trabajo generadas")
        return ot_generadas

    def generar_ot_preventivas(self, df_datos):
        """Genera OT preventivas programadas"""
        ot_preventivas = []
        compresores = df_datos['Compresor_ID'].unique()

        for compresor in compresores:
            # Simular OT preventiva cada 90 días
            if np.random.random() < 0.3:  # 30% probabilidad de OT preventiva
                conocimiento = self.base_conocimientos['Mantenimiento_Preventivo']

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ot_id = f"FP1-{compresor}-PREV-{timestamp}"

                ot_preventiva = {
                    'ot_id': ot_id,
                    'fecha_creacion': datetime.now().isoformat(),
                    'fecha_objetivo': (datetime.now() + timedelta(days=7)).isoformat(),
                    'compresor_id': compresor,
                    'tipo_mantenimiento': 'Preventivo',
                    'prioridad': 'Media',
                    'estado': 'Programada',
                    'titulo': 'Mantenimiento preventivo programado',
                    'diagnostico_principal': 'Mantenimiento_Preventivo',
                    'explicacion_detallada': f"Mantenimiento preventivo programado para {compresor} según cronograma establecido.",
                    'acciones_recomendadas': conocimiento['acciones'],
                    'tiempo_estimado_horas': conocimiento['tiempo_estimado'],
                    'especialista_requerido': conocimiento['especialista'],
                    'repuestos_sugeridos': conocimiento['repuestos'],
                    'metodo_deteccion': 'Programado',
                    'confianza_diagnostico': 1.0
                }

                ot_preventivas.append(ot_preventiva)

        return ot_preventivas

    def guardar_ot_generadas(self):
        """Guarda las OT generadas en archivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_ot = f'output/reports/ordenes_trabajo_{timestamp}.json'

        with open(archivo_ot, 'w', encoding='utf-8') as f:
            json.dump(self.ot_generadas, f, indent=2, ensure_ascii=False, default=str)

        # También generar CSV para fácil lectura
        df_ot = pd.DataFrame(self.ot_generadas)
        archivo_csv = f'output/datasets/ordenes_trabajo_{timestamp}.csv'
        df_ot.to_csv(archivo_csv, index=False, encoding='utf-8')

        self.logger.info(f"💾 OT guardadas en: {archivo_ot} y {archivo_csv}")

    def generar_reporte_ot(self):
        """Genera reporte de las OT generadas"""
        if not self.ot_generadas:
            return "No hay órdenes de trabajo generadas"

        # Estadísticas
        total_ot = len(self.ot_generadas)
        por_tipo = pd.Series([ot['tipo_mantenimiento'] for ot in self.ot_generadas]).value_counts()
        por_prioridad = pd.Series([ot['prioridad'] for ot in self.ot_generadas]).value_counts()
        por_compresor = pd.Series([ot['compresor_id'] for ot in self.ot_generadas]).value_counts()

        reporte = f"""
REPORTE DE ÓRDENES DE TRABAJO GENERADAS
=======================================
Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

RESUMEN GENERAL:
- Total OT generadas: {total_ot}

DISTRIBUCIÓN POR TIPO:
{por_tipo.to_string()}

DISTRIBUCIÓN POR PRIORIDAD:
{por_prioridad.to_string()}

DISTRIBUCIÓN POR COMPRESOR:
{por_compresor.to_string()}

OT CRÍTICAS Y ALTAS:
{self.listar_ot_criticas()}

PRÓXIMAS ACCIONES RECOMENDADAS:
1. Revisar y aprobar OT críticas inmediatamente
2. Asignar recursos para OT de alta prioridad
3. Programar OT preventivas según disponibilidad
4. Validar diagnósticos con personal especializado
        """.strip()

        return reporte

    def listar_ot_criticas(self):
        """Lista las OT críticas y de alta prioridad"""
        ot_importantes = [ot for ot in self.ot_generadas if ot['prioridad'] in ['Crítica', 'Alta']]

        if not ot_importantes:
            return "No hay OT críticas o de alta prioridad"

        lineas = []
        for ot in ot_importantes:
            lineas.append(f"- {ot['ot_id']}: {ot['titulo']} ({ot['compresor_id']}) - {ot['prioridad']}")

        return "\n".join(lineas)

# FUNCIÓN DE PRUEBA
def probar_generador():
    """Función de prueba del generador de OT"""
    print("🧪 Probando Generador de Órdenes de Trabajo...")

    # Datos de prueba
    datos_prueba = pd.DataFrame({
        'Compresor_ID': ['C1', 'C2', 'C3'],
        'THD_Total': [8.5, 2.1, 3.2],  # C1 tiene THD alto
        'Factor_Potencia': [0.85, 0.6, 0.88],  # C2 tiene factor bajo
        'Vibracion_RMS_Vectorial': [3.2, 4.1, 9.5],  # C3 tiene vibración alta
        'Potencia_Activa': [45, 42, 38],
        'Timestamp': pd.date_range('2024-01-01', periods=3, freq='1H')
    })

    anomalias_prueba = np.array([True, True, True])  # Todas son anomalías

    generador = GeneradorOrdenesTrabajo()
    ot_generadas = generador.procesar_anomalias_y_generar_ot(datos_prueba, anomalias_prueba)

    print(f"✅ Generadas {len(ot_generadas)} OT de prueba")

    # Mostrar primera OT como ejemplo
    if ot_generadas:
        print("\n📋 EJEMPLO DE OT GENERADA:")
        print(f"ID: {ot_generadas[0]['ot_id']}")
        print(f"Tipo: {ot_generadas[0]['tipo_mantenimiento']}")
        print(f"Prioridad: {ot_generadas[0]['prioridad']}")
        print(f"Descripción: {ot_generadas[0]['titulo']}")

    return generador

if __name__ == "__main__":
    probar_generador()
