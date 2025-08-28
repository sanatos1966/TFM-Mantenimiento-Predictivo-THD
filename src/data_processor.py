
"""
Procesador de Datos para Sistema de Mantenimiento Predictivo
==========================================================

Maneja la carga y procesamiento de nuevos datos en múltiples formatos:
CSV, XLSX, PDF para el sistema de mantenimiento predictivo.

Autor: Antonio Cantos & Renzo Chavez(TFM)  
Fecha: Enero 2025
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import PyPDF2
import pdfplumber
from openpyxl import load_workbook
import logging
import re
from typing import Dict, List, Tuple, Union
import chardet
import warnings
warnings.filterwarnings('ignore')

class ProcesadorDatos:
    """
    Procesador de datos que maneja múltiples formatos de entrada
    y los convierte al formato estándar del sistema.
    """

    def __init__(self, config_path="config/config.json"):
        """
        Inicializa el procesador de datos.

        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.cargar_configuracion()
        self.configurar_logging()
        self.formatos_soportados = ['.csv', '.xlsx', '.xls', '.pdf']
        self.columnas_requeridas = [
            'THD_I_L1(%)', 'THD_V_L1(%)', 'Factor_Potencia',
            'Corriente_L1(A)', 'Vibracion_Axial', 'Compresor_ID'
        ]

    def cargar_configuracion(self):
        """Carga la configuración desde el archivo JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo de configuración {self.config_path}")
            # Configuración por defecto
            self.config = {
                "procesamiento": {
                    "encoding_csv": "utf-8",
                    "separador_csv": ",",
                    "validacion_datos": True,
                    "limpieza_automatica": True
                }
            }

    def configurar_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def detectar_encoding(self, ruta_archivo: str) -> str:
        """
        Detecta el encoding de un archivo de texto.

        Args:
            ruta_archivo (str): Ruta al archivo

        Returns:
            str: Encoding detectado
        """
        try:
            with open(ruta_archivo, 'rb') as f:
                raw_data = f.read(10000)  # Leer primeros 10KB
                resultado = chardet.detect(raw_data)
                encoding = resultado['encoding']
                confidence = resultado['confidence']

                if confidence > 0.7:
                    self.logger.info(f"Encoding detectado: {encoding} (confianza: {confidence:.2f})")
                    return encoding
                else:
                    self.logger.warning(f"Baja confianza en encoding detectado, usando UTF-8")
                    return 'utf-8'

        except Exception as e:
            self.logger.error(f"Error detectando encoding: {str(e)}")
            return 'utf-8'

    def procesar_archivo_csv(self, ruta_archivo: str) -> pd.DataFrame:
        """
        Procesa un archivo CSV y retorna un DataFrame.

        Args:
            ruta_archivo (str): Ruta al archivo CSV

        Returns:
            pd.DataFrame: Datos procesados
        """
        try:
            # Detectar encoding automáticamente
            encoding = self.detectar_encoding(ruta_archivo)

            # Intentar diferentes separadores
            separadores = [',', ';', '\t', '|']

            for sep in separadores:
                try:
                    df = pd.read_csv(ruta_archivo, 
                                   encoding=encoding, 
                                   separator=sep,
                                   low_memory=False)

                    if len(df.columns) > 1:  # Si hay múltiples columnas, el separador es correcto
                        self.logger.info(f"CSV cargado exitosamente con separador '{sep}': {df.shape}")
                        break

                except Exception as e:
                    continue
            else:
                raise ValueError("No se pudo determinar el separador correcto del CSV")

            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip()

            return df

        except Exception as e:
            self.logger.error(f"Error procesando CSV {ruta_archivo}: {str(e)}")
            raise

    def procesar_archivo_excel(self, ruta_archivo: str) -> pd.DataFrame:
        """
        Procesa un archivo Excel (XLSX/XLS) y retorna un DataFrame.

        Args:
            ruta_archivo (str): Ruta al archivo Excel

        Returns:
            pd.DataFrame: Datos procesados
        """
        try:
            # Intentar leer múltiples hojas
            excel_file = pd.ExcelFile(ruta_archivo)
            hojas = excel_file.sheet_names

            self.logger.info(f"Hojas disponibles en Excel: {hojas}")

            # Priorizar hojas con nombres relevantes
            hojas_prioritarias = ['datos', 'data', 'compresores', 'mediciones', 'sheet1']

            hoja_seleccionada = None
            for hoja_prio in hojas_prioritarias:
                for hoja in hojas:
                    if hoja_prio.lower() in hoja.lower():
                        hoja_seleccionada = hoja
                        break
                if hoja_seleccionada:
                    break

            if not hoja_seleccionada:
                hoja_seleccionada = hojas[0]  # Usar primera hoja por defecto

            self.logger.info(f"Procesando hoja: {hoja_seleccionada}")

            df = pd.read_excel(ruta_archivo, sheet_name=hoja_seleccionada)

            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip()

            self.logger.info(f"Excel cargado exitosamente: {df.shape}")

            return df

        except Exception as e:
            self.logger.error(f"Error procesando Excel {ruta_archivo}: {str(e)}")
            raise

    def procesar_archivo_pdf(self, ruta_archivo: str) -> pd.DataFrame:
        """
        Procesa un archivo PDF extrayendo tablas de datos.

        Args:
            ruta_archivo (str): Ruta al archivo PDF

        Returns:
            pd.DataFrame: Datos procesados
        """
        try:
            datos_extraidos = []

            # Método 1: Usar pdfplumber para extraer tablas
            try:
                with pdfplumber.open(ruta_archivo) as pdf:
                    for i, page in enumerate(pdf.pages):
                        # Extraer tablas de la página
                        tables = page.extract_tables()

                        for table in tables:
                            if table and len(table) > 1:  # Al menos header + 1 fila
                                # Convertir tabla a DataFrame
                                df_tabla = pd.DataFrame(table[1:], columns=table[0])

                                # Limpiar datos
                                df_tabla = self.limpiar_datos_pdf(df_tabla)

                                if not df_tabla.empty:
                                    datos_extraidos.append(df_tabla)

                self.logger.info(f"Tablas extraídas del PDF: {len(datos_extraidos)}")

            except Exception as e:
                self.logger.warning(f"Error con pdfplumber: {str(e)}, intentando PyPDF2")

                # Método 2: Fallback con PyPDF2 para texto simple
                with open(ruta_archivo, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)

                    texto_completo = ""
                    for page in pdf_reader.pages:
                        texto_completo += page.extract_text()

                    # Intentar extraer datos tabulares del texto
                    df_texto = self.extraer_datos_de_texto(texto_completo)
                    if not df_texto.empty:
                        datos_extraidos.append(df_texto)

            # Combinar todos los datos extraídos
            if datos_extraidos:
                df_final = pd.concat(datos_extraidos, ignore_index=True)
                self.logger.info(f"PDF procesado exitosamente: {df_final.shape}")
                return df_final
            else:
                raise ValueError("No se pudieron extraer datos tabulares del PDF")

        except Exception as e:
            self.logger.error(f"Error procesando PDF {ruta_archivo}: {str(e)}")
            raise

    def limpiar_datos_pdf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia datos extraídos de PDF.

        Args:
            df (pd.DataFrame): DataFrame con datos de PDF

        Returns:
            pd.DataFrame: DataFrame limpio
        """
        try:
            # Eliminar filas y columnas completamente vacías
            df = df.dropna(how='all').dropna(axis=1, how='all')

            # Limpiar nombres de columnas
            df.columns = df.columns.astype(str).str.strip()

            # Limpiar celdas de datos
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(['None', 'nan', ''], np.nan)

            # Intentar convertir columnas numéricas
            for col in df.columns:
                try:
                    # Limpiar caracteres no numéricos comunes en PDFs
                    df_temp = df[col].astype(str).str.replace(',', '.')
                    df_temp = df_temp.str.replace('[^\d.-]', '', regex=True)
                    df_temp = pd.to_numeric(df_temp, errors='ignore')

                    if df_temp.dtype in ['int64', 'float64']:
                        df[col] = df_temp

                except Exception:
                    continue

            return df

        except Exception as e:
            self.logger.error(f"Error limpiando datos PDF: {str(e)}")
            return df

    def extraer_datos_de_texto(self, texto: str) -> pd.DataFrame:
        """
        Extrae datos tabulares de texto plano.

        Args:
            texto (str): Texto extraído del PDF

        Returns:
            pd.DataFrame: Datos extraídos
        """
        try:
            lineas = texto.split('\n')
            datos_numericos = []

            # Buscar líneas que contengan múltiples números (posibles datos)
            patron_numerico = r'[-+]?\d*\.?\d+'

            for linea in lineas:
                numeros = re.findall(patron_numerico, linea)

                # Si la línea tiene múltiples números, puede ser una fila de datos
                if len(numeros) >= 3:
                    try:
                        fila_numerica = [float(num) for num in numeros]
                        datos_numericos.append(fila_numerica)
                    except ValueError:
                        continue

            if datos_numericos:
                # Determinar número de columnas más común
                longitudes = [len(fila) for fila in datos_numericos]
                num_columnas = max(set(longitudes), key=longitudes.count)

                # Filtrar filas con el número correcto de columnas
                datos_filtrados = [fila for fila in datos_numericos if len(fila) == num_columnas]

                if datos_filtrados:
                    # Crear DataFrame con nombres genéricos de columnas
                    columnas = [f'Col_{i+1}' for i in range(num_columnas)]
                    df = pd.DataFrame(datos_filtrados, columns=columnas)

                    return df

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error extrayendo datos de texto: {str(e)}")
            return pd.DataFrame()

    def mapear_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mapea columnas del archivo a las columnas estándar del sistema.

        Args:
            df (pd.DataFrame): DataFrame con datos originales

        Returns:
            pd.DataFrame: DataFrame con columnas mapeadas
        """
        try:
            # Diccionario de mapeo de nombres alternativos
            mapeo_columnas = {
                'thd_i_l1': 'THD_I_L1(%)',
                'thd_v_l1': 'THD_V_L1(%)', 
                'factor_potencia': 'Factor_Potencia',
                'corriente_l1': 'Corriente_L1(A)',
                'vibracion_axial': 'Vibracion_Axial',
                'compresor_id': 'Compresor_ID',
                'compresor': 'Compresor_ID',
                'id_compresor': 'Compresor_ID',
                'timestamp': 'Timestamp',
                'fecha': 'Timestamp',
                'time': 'Timestamp'
            }

            # Crear mapeo inverso con múltiples variaciones
            mapeo_expandido = {}
            for key, value in mapeo_columnas.items():
                # Variaciones del nombre
                variaciones = [
                    key,
                    key.upper(),
                    key.lower(),
                    key.replace('_', ' '),
                    key.replace('_', ''),
                    key.replace(' ', '_')
                ]

                for variacion in variaciones:
                    mapeo_expandido[variacion] = value

            # Intentar mapear columnas existentes
            df_mapeado = df.copy()
            columnas_originales = list(df.columns)

            for col_original in columnas_originales:
                col_limpia = col_original.strip().lower()

                # Buscar coincidencia exacta
                if col_limpia in mapeo_expandido:
                    nuevo_nombre = mapeo_expandido[col_limpia]
                    df_mapeado = df_mapeado.rename(columns={col_original: nuevo_nombre})
                    self.logger.info(f"Columna mapeada: {col_original} -> {nuevo_nombre}")
                    continue

                # Buscar coincidencia parcial
                for patron, nuevo_nombre in mapeo_expandido.items():
                    if patron in col_limpia or col_limpia in patron:
                        df_mapeado = df_mapeado.rename(columns={col_original: nuevo_nombre})
                        self.logger.info(f"Columna mapeada (parcial): {col_original} -> {nuevo_nombre}")
                        break

            return df_mapeado

        except Exception as e:
            self.logger.error(f"Error mapeando columnas: {str(e)}")
            return df

    def validar_datos(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Valida que los datos cumplan con los requisitos del sistema.

        Args:
            df (pd.DataFrame): DataFrame a validar

        Returns:
            Tuple[bool, List[str]]: (Es válido, Lista de errores)
        """
        errores = []

        try:
            # Verificar que el DataFrame no esté vacío
            if df.empty:
                errores.append("El archivo no contiene datos")
                return False, errores

            # Verificar columnas mínimas requeridas
            columnas_presentes = set(df.columns)
            columnas_requeridas_set = set(self.columnas_requeridas)
            columnas_faltantes = columnas_requeridas_set - columnas_presentes

            if columnas_faltantes:
                errores.append(f"Faltan columnas requeridas: {list(columnas_faltantes)}")

            # Verificar tipos de datos numéricos
            columnas_numericas = ['THD_I_L1(%)', 'THD_V_L1(%)', 'Factor_Potencia', 
                                'Corriente_L1(A)', 'Vibracion_Axial']

            for col in columnas_numericas:
                if col in df.columns:
                    try:
                        pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        errores.append(f"Columna {col} no contiene datos numéricos válidos")

            # Verificar rangos de valores
            rangos_validos = {
                'THD_I_L1(%)': (0, 100),
                'THD_V_L1(%)': (0, 100),
                'Factor_Potencia': (0, 1),
                'Corriente_L1(A)': (0, 1000),
                'Vibracion_Axial': (0, 100)
            }

            for col, (min_val, max_val) in rangos_validos.items():
                if col in df.columns:
                    valores_numericos = pd.to_numeric(df[col], errors='coerce')
                    valores_fuera_rango = (valores_numericos < min_val) | (valores_numericos > max_val)

                    if valores_fuera_rango.any():
                        count_fuera = valores_fuera_rango.sum()
                        errores.append(f"Columna {col}: {count_fuera} valores fuera del rango válido [{min_val}, {max_val}]")

            # Verificar porcentaje de datos faltantes
            porcentaje_faltantes = df.isnull().sum() / len(df) * 100
            columnas_muchos_faltantes = porcentaje_faltantes[porcentaje_faltantes > 50]

            if not columnas_muchos_faltantes.empty:
                errores.append(f"Columnas con >50% datos faltantes: {dict(columnas_muchos_faltantes)}")

            es_valido = len(errores) == 0

            if es_valido:
                self.logger.info("Validación de datos exitosa")
            else:
                self.logger.warning(f"Errores de validación: {errores}")

            return es_valido, errores

        except Exception as e:
            self.logger.error(f"Error en validación: {str(e)}")
            return False, [f"Error durante la validación: {str(e)}"]

    def limpiar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y prepara los datos para el sistema.

        Args:
            df (pd.DataFrame): DataFrame original

        Returns:
            pd.DataFrame: DataFrame limpio
        """
        try:
            df_limpio = df.copy()

            # 1. Eliminar duplicados
            duplicados_antes = len(df_limpio)
            df_limpio = df_limpio.drop_duplicates()
            duplicados_eliminados = duplicados_antes - len(df_limpio)

            if duplicados_eliminados > 0:
                self.logger.info(f"Eliminados {duplicados_eliminados} registros duplicados")

            # 2. Limpiar columnas numéricas
            columnas_numericas = ['THD_I_L1(%)', 'THD_V_L1(%)', 'Factor_Potencia', 
                                'Corriente_L1(A)', 'Vibracion_Axial']

            for col in columnas_numericas:
                if col in df_limpio.columns:
                    # Convertir a numérico, NaN para valores inválidos
                    df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')

                    # Imputar valores faltantes con la mediana
                    if df_limpio[col].isnull().any():
                        mediana = df_limpio[col].median()
                        valores_faltantes = df_limpio[col].isnull().sum()
                        df_limpio[col].fillna(mediana, inplace=True)
                        self.logger.info(f"Imputados {valores_faltantes} valores faltantes en {col} con mediana: {mediana:.3f}")

            # 3. Limpiar columnas categóricas
            if 'Compresor_ID' in df_limpio.columns:
                df_limpio['Compresor_ID'] = df_limpio['Compresor_ID'].astype(str).str.strip()
                df_limpio['Compresor_ID'] = df_limpio['Compresor_ID'].replace(['nan', 'None', ''], 'Compresor_Desconocido')

            # 4. Añadir timestamp si no existe
            if 'Timestamp' not in df_limpio.columns:
                # Crear timestamps secuenciales cada minuto
                base_time = datetime.now() - timedelta(minutes=len(df_limpio))
                timestamps = [base_time + timedelta(minutes=i) for i in range(len(df_limpio))]
                df_limpio['Timestamp'] = timestamps
                self.logger.info("Timestamps generados automáticamente")

            # 5. Filtrar valores extremos (outliers)
            for col in columnas_numericas:
                if col in df_limpio.columns:
                    Q1 = df_limpio[col].quantile(0.25)
                    Q3 = df_limpio[col].quantile(0.75)
                    IQR = Q3 - Q1

                    limite_inferior = Q1 - 3 * IQR  # Más permisivo que 1.5*IQR
                    limite_superior = Q3 + 3 * IQR

                    outliers = (df_limpio[col] < limite_inferior) | (df_limpio[col] > limite_superior)

                    if outliers.any():
                        # Reemplazar outliers con valores límite en lugar de eliminar
                        df_limpio.loc[df_limpio[col] < limite_inferior, col] = limite_inferior
                        df_limpio.loc[df_limpio[col] > limite_superior, col] = limite_superior
                        self.logger.info(f"Corregidos {outliers.sum()} outliers en {col}")

            self.logger.info(f"Datos limpiados: {len(df_limpio)} registros finales")

            return df_limpio

        except Exception as e:
            self.logger.error(f"Error limpiando datos: {str(e)}")
            return df

    def procesar_archivo(self, ruta_archivo: str) -> Dict:
        """
        Función principal que procesa un archivo de cualquier formato soportado.

        Args:
            ruta_archivo (str): Ruta al archivo a procesar

        Returns:
            Dict: Resultado del procesamiento con datos y metadata
        """
        resultado = {
            'exito': False,
            'archivo': ruta_archivo,
            'formato': None,
            'datos': None,
            'filas_procesadas': 0,
            'columnas_procesadas': 0,
            'errores': [],
            'advertencias': [],
            'timestamp_procesamiento': datetime.now().isoformat()
        }

        try:
            # Verificar que el archivo existe
            if not os.path.exists(ruta_archivo):
                resultado['errores'].append(f"Archivo no encontrado: {ruta_archivo}")
                return resultado

            # Determinar formato del archivo
            _, extension = os.path.splitext(ruta_archivo.lower())
            resultado['formato'] = extension

            if extension not in self.formatos_soportados:
                resultado['errores'].append(f"Formato no soportado: {extension}")
                return resultado

            self.logger.info(f"Procesando archivo: {ruta_archivo} (formato: {extension})")

            # Procesar según el formato
            if extension == '.csv':
                df = self.procesar_archivo_csv(ruta_archivo)
            elif extension in ['.xlsx', '.xls']:
                df = self.procesar_archivo_excel(ruta_archivo)
            elif extension == '.pdf':
                df = self.procesar_archivo_pdf(ruta_archivo)
            else:
                raise ValueError(f"Formato {extension} no implementado")

            # Mapear columnas al estándar del sistema
            df_mapeado = self.mapear_columnas(df)

            # Validar datos
            es_valido, errores_validacion = self.validar_datos(df_mapeado)
            resultado['errores'].extend(errores_validacion)

            if not es_valido and self.config.get('procesamiento', {}).get('validacion_estricta', False):
                return resultado

            # Limpiar datos si está habilitado
            if self.config.get('procesamiento', {}).get('limpieza_automatica', True):
                df_final = self.limpiar_datos(df_mapeado)
            else:
                df_final = df_mapeado

            # Validación final
            if df_final.empty:
                resultado['errores'].append("No quedaron datos después del procesamiento")
                return resultado

            # Guardar resultados
            resultado['datos'] = df_final
            resultado['filas_procesadas'] = len(df_final)
            resultado['columnas_procesadas'] = len(df_final.columns)
            resultado['exito'] = True

            self.logger.info(f"Archivo procesado exitosamente: {resultado['filas_procesadas']} filas, {resultado['columnas_procesadas']} columnas")

        except Exception as e:
            error_msg = f"Error procesando archivo: {str(e)}"
            resultado['errores'].append(error_msg)
            self.logger.error(error_msg)

        return resultado

    def procesar_multiples_archivos(self, rutas_archivos: List[str]) -> Dict:
        """
        Procesa múltiples archivos y combina los resultados.

        Args:
            rutas_archivos (List[str]): Lista de rutas de archivos

        Returns:
            Dict: Resultado combinado del procesamiento
        """
        resultado_combinado = {
            'exito': False,
            'archivos_procesados': 0,
            'archivos_fallidos': 0,
            'datos_combinados': None,
            'total_filas': 0,
            'errores_por_archivo': {},
            'timestamp_procesamiento': datetime.now().isoformat()
        }

        try:
            datos_todos = []

            for ruta_archivo in rutas_archivos:
                resultado = self.procesar_archivo(ruta_archivo)

                if resultado['exito']:
                    datos_todos.append(resultado['datos'])
                    resultado_combinado['archivos_procesados'] += 1
                else:
                    resultado_combinado['archivos_fallidos'] += 1
                    resultado_combinado['errores_por_archivo'][ruta_archivo] = resultado['errores']

            # Combinar todos los datos
            if datos_todos:
                df_combinado = pd.concat(datos_todos, ignore_index=True)

                # Limpiar datos combinados
                df_final = self.limpiar_datos(df_combinado)

                resultado_combinado['datos_combinados'] = df_final
                resultado_combinado['total_filas'] = len(df_final)
                resultado_combinado['exito'] = True

                self.logger.info(f"Procesamiento múltiple completado: {resultado_combinado['archivos_procesados']} archivos exitosos, {resultado_combinado['total_filas']} filas totales")

            else:
                self.logger.warning("Ningún archivo se procesó exitosamente")

        except Exception as e:
            self.logger.error(f"Error en procesamiento múltiple: {str(e)}")
            resultado_combinado['error_general'] = str(e)

        return resultado_combinado

    def guardar_datos_procesados(self, datos: pd.DataFrame, ruta_salida: str, formato_salida: str = 'csv'):
        """
        Guarda los datos procesados en el formato especificado.

        Args:
            datos (pd.DataFrame): Datos a guardar
            ruta_salida (str): Ruta donde guardar los datos
            formato_salida (str): Formato de salida ('csv', 'xlsx', 'json')
        """
        try:
            # Crear directorio si no existe
            directorio = os.path.dirname(ruta_salida)
            if directorio:
                os.makedirs(directorio, exist_ok=True)

            if formato_salida.lower() == 'csv':
                datos.to_csv(ruta_salida, index=False, encoding='utf-8')
            elif formato_salida.lower() == 'xlsx':
                datos.to_excel(ruta_salida, index=False)
            elif formato_salida.lower() == 'json':
                datos.to_json(ruta_salida, orient='records', date_format='iso', force_ascii=False, indent=2)
            else:
                raise ValueError(f"Formato de salida no soportado: {formato_salida}")

            self.logger.info(f"Datos guardados exitosamente en {ruta_salida}")

        except Exception as e:
            self.logger.error(f"Error guardando datos: {str(e)}")
            raise


# Función auxiliar para uso independiente
def procesar_archivo_unico(ruta_archivo: str, config_path: str = "config/config.json") -> Dict:
    """
    Función auxiliar para procesar un solo archivo.

    Args:
        ruta_archivo (str): Ruta al archivo
        config_path (str): Ruta a la configuración

    Returns:
        Dict: Resultado del procesamiento
    """
    procesador = ProcesadorDatos(config_path)
    return procesador.procesar_archivo(ruta_archivo)


if __name__ == "__main__":
    # Ejemplo de uso
    print("Procesador de Datos - Sistema de Mantenimiento Predictivo")
    print("=" * 55)

    # Crear instancia del procesador
    procesador = ProcesadorDatos()

    print(f"Formatos soportados: {procesador.formatos_soportados}")
    print(f"Columnas requeridas: {procesador.columnas_requeridas}")
