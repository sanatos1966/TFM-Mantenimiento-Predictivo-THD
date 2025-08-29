# watch_new_raw.py
import os, time, json
from pathlib import Path
from src.data_processor import ProcesadorDatos
from src.tfm_pipeline import SistemaMantenimientoPredictivo
from src.ot_generator import GeneradorOrdenesTrabajo
from src.learning_system import SistemaAprendizajeContinuo

RAW_DIR = Path("data/raw")
STATE = Path("output/_processed_files.json")
CFG = "config/config.json"
SLEEP_SEC = 300  # 5 minutos

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"processed": []}

def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def main():
    state = load_state()
    proc = ProcesadorDatos(CFG)
    sist = SistemaMantenimientoPredictivo(CFG)
    gen  = GeneradorOrdenesTrabajo(CFG)
    apr  = SistemaAprendizajeContinuo(CFG)

    print("Watcher activo sobre data/raw/. Procesará archivos nuevos .csv/.xlsx/.pdf")

    while True:
        try:
            candidates = [p for p in RAW_DIR.glob("*") if p.suffix.lower() in [".csv",".xlsx",".pdf"]]
            new_files = [str(p) for p in candidates if str(p) not in state["processed"]]

            if new_files:
                print(f"Archivos nuevos detectados: {new_files}")
                res = proc.procesar_multiples_archivos(new_files)
                if res.get("exito"):
                    datos = res["datos_combinados"]
                    anom = sist.detectar_anomalias_tiempo_real(datos)
                    ots  = gen.generar_ordenes_multiples(anom)
                    apr.aprendizaje_incremental(datos)
                    print(f"OK - filas={res['total_filas']}, anomalias={anom['anomalias_detectadas']}, OTs={len(ots)}")
                    state["processed"].extend(new_files)
                    save_state(state)
                else:
                    print(f"Fallo procesando: {res.get('errores_por_archivo')}")
            time.sleep(SLEEP_SEC)
        except KeyboardInterrupt:
            print("Watcher detenido por usuario.")
            break
        except Exception as e:
            print(f"Error en ciclo: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
