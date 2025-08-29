# tests/test_data_processor.py
import pandas as pd
from pathlib import Path
from src.data_processor import ProcesadorDatos

def _df_minimo_ok():
    # Columnas EXACTAS requeridas por el validador
    return pd.DataFrame(
        {
            "THD_I_L1(%)": [3.2, 4.1],
            "THD_V_L1(%)": [2.5, 2.8],
            "Factor_Potencia": [0.92, 0.95],
            "Corriente_L1(A)": [12.3, 11.8],
            "Vibracion_Axial": [1.2, 1.5],
            "Compresor_ID": ["C001", "C001"],
        }
    )

def test_csv_autodeteccion_separador(tmp_path: Path):
    df = _df_minimo_ok()
    # Guardamos con separador ';' para forzar autodetección
    csv_path = tmp_path / "datos_semicolon.csv"
    df.to_csv(csv_path, index=False, sep=";")

    proc = ProcesadorDatos()  # usa config por defecto
    res = proc.procesar_archivo(str(csv_path))

    assert res["exito"] is True
    assert res["filas_procesadas"] == len(df)
    assert res["formato"] == ".csv"
    assert set(res["datos"].columns) >= set(df.columns)

def test_excel_xlsx_openpyxl(tmp_path: Path):
    df = _df_minimo_ok()
    xlsx_path = tmp_path / "datos.xlsx"
    # Guardar con engine openpyxl (coherente con requirements Opción A)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    proc = ProcesadorDatos()
    res = proc.procesar_archivo(str(xlsx_path))

    assert res["exito"] is True
    assert res["filas_procesadas"] == len(df)
    assert res["formato"] == ".xlsx"
    assert set(res["datos"].columns) >= set(df.columns)

def test_validacion_estricta_corta_flujo(tmp_path: Path):
    # Config temporal con validación estricta activada
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.json"

    cfg_path.write_text(
        """
        {
          "procesamiento": {
            "encoding_csv": "utf-8",
            "separador_csv": ",",
            "validacion_datos": true,
            "validacion_estricta": true,
            "limpieza_automatica": true
          }
        }
        """,
        encoding="utf-8",
    )

    # CSV con una columna requerida ausente para provocar error
    df = _df_minimo_ok().drop(columns=["Vibracion_Axial"])
    csv_path = tmp_path / "datos_incompletos.csv"
    df.to_csv(csv_path, index=False)

    proc = ProcesadorDatos(config_path=str(cfg_path))
    res = proc.procesar_archivo(str(csv_path))

    assert res["exito"] is False
    # Debe reportar la columna faltante
    assert any("Faltan columnas requeridas" in e for e in res["errores"])
