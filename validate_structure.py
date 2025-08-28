
import os
import sys
import json
from pathlib import Path
import pandas as pd

def validate_project_structure():
    """
    Validate the complete TFM project structure and files
    """
    print("=== VALIDACIÓN ESTRUCTURA PROYECTO TFM ===\n")

    # Expected project structure
    expected_structure = {
        'files': [
            'TFM_Pipeline_Real_Final_20250826_1951.py',
            'README.md',
            'LICENSE',
            '.gitignore'
        ],
        'directories': {
            'src': ['__init__.py'],
            'src/utils': ['__init__.py'],
            'config': ['config.json'],
            'data': ['datos_completos_tfm.csv', 'datos_completos_tfm.xlsx'],
            'tests': ['__init__.py'],
            'docs': [],
            'output': []
        }
    }

    # Check root files
    print("📁 Archivos raíz:")
    for file in expected_structure['files']:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")

    # Check directories and their contents
    print("\n📂 Directorios y contenido:")
    for dir_path, files in expected_structure['directories'].items():
        dir_exists = os.path.exists(dir_path)
        status = "✅" if dir_exists else "❌"
        print(f"  {status} {dir_path}/")

        if dir_exists and files:
            for file in files:
                file_path = os.path.join(dir_path, file)
                file_exists = os.path.exists(file_path)
                file_status = "✅" if file_exists else "❌"
                print(f"    {file_status} {file}")

    # Validate configuration file
    print("\n🔧 Validación configuración:")
    config_path = 'config/config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Check ML parameters
            if 'ml_parameters' in config:
                print("  ✅ Parámetros ML encontrados")

                # Check Isolation Forest parameters
                if 'isolation_forest' in config['ml_parameters']:
                    if_params = config['ml_parameters']['isolation_forest']
                    expected_contamination = 0.004319
                    actual_contamination = if_params.get('contamination', 0)

                    if abs(actual_contamination - expected_contamination) < 0.000001:
                        print(f"  ✅ Contamination correcto: {actual_contamination}")
                    else:
                        print(f"  ❌ Contamination incorrecto: {actual_contamination} (esperado: {expected_contamination})")

                    expected_estimators = 200
                    actual_estimators = if_params.get('n_estimators', 0)

                    if actual_estimators == expected_estimators:
                        print(f"  ✅ N_estimators correcto: {actual_estimators}")
                    else:
                        print(f"  ❌ N_estimators incorrecto: {actual_estimators} (esperado: {expected_estimators})")

                # Check DBSCAN parameters
                if 'dbscan' in config['ml_parameters']:
                    dbscan_params = config['ml_parameters']['dbscan']
                    expected_eps = 1.2
                    actual_eps = dbscan_params.get('eps', 0)

                    if abs(actual_eps - expected_eps) < 0.1:
                        print(f"  ✅ EPS correcto: {actual_eps}")
                    else:
                        print(f"  ❌ EPS incorrecto: {actual_eps} (esperado: {expected_eps})")
            else:
                print("  ❌ No se encontraron parámetros ML")

        except Exception as e:
            print(f"  ❌ Error leyendo configuración: {e}")
    else:
        print("  ❌ Archivo config.json no encontrado")

    # Validate data files
    print("\n📊 Validación archivos de datos:")
    data_files = ['data/datos_completos_tfm.csv', 'data/datos_completos_tfm.xlsx']

    for data_file in data_files:
        if os.path.exists(data_file):
            try:
                if data_file.endswith('.csv'):
                    df = pd.read_csv(data_file)
                else:
                    df = pd.read_excel(data_file)

                print(f"  ✅ {data_file}: {len(df)} registros, {len(df.columns)} columnas")

                # Check for expected columns
                expected_cols = ['timestamp', 'compressor', 'vibration', 'current', 'thd']
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    print(f"    ⚠️  Columnas faltantes: {missing_cols}")
                else:
                    print("    ✅ Todas las columnas esperadas presentes")

            except Exception as e:
                print(f"  ❌ Error leyendo {data_file}: {e}")
        else:
            print(f"  ❌ {data_file} no encontrado")

    # Check Python modules can be imported
    print("\n🐍 Validación módulos Python:")
    try:
        sys.path.insert(0, '.')
        import src
        print("  ✅ Módulo 'src' importado correctamente")
    except Exception as e:
        print(f"  ❌ Error importando 'src': {e}")

    try:
        import src.utils
        print("  ✅ Módulo 'src.utils' importado correctamente")
    except Exception as e:
        print(f"  ❌ Error importando 'src.utils': {e}")

    print("\n=== VALIDACIÓN COMPLETA ===")

if __name__ == "__main__":
    validate_project_structure()
