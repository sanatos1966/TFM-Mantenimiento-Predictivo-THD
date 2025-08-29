# tests/test_install.py
"""
Verificación básica de instalación del sistema TFM.
Ejecuta imports críticos y muestra confirmación si todo está correcto.
"""
import sys

try:
    from src.tfm_pipeline import SistemaMantenimientoPredictivo
    from src.data_processor import ProcesadorDatos
    from src.ot_generator import GeneradorOrdenesTrabajo
    from src.learning_system import SistemaAprendizajeContinuo
except Exception as e:
    print("❌ Error en la verificación")
    print(f"   Detalle: {type(e).__name__} -> {e}")
    sys.exit(1)
else:
    print("✅ Verificación completada con éxito")
    print("   - Imports OK")
    print("   - Estructura del repo detectada correctamente")
    sys.exit(0)
