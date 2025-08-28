#!/usr/bin/env python3
"""
Setup script for TFM Predictive Maintenance System
Author: Antonio (TFM)
Date: January 2025
"""

from setuptools import setup, find_packages
import os
import sys

# Verify Python version
if sys.version_info < (3, 8):
    raise RuntimeError("Python 3.8 or higher is required")

# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Sistema de Mantenimiento Predictivo Industrial - TFM Antonio"

# Read requirements
def read_requirements():
    requirements = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        # Fallback requirements
        requirements = [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "joblib>=1.1.0",
            "openpyxl>=3.0.9",
            "PyPDF2>=2.10.0",
            "pdfplumber>=0.7.0",
            "chardet>=5.0.0",
            "tqdm>=4.62.0"
        ]
    return requirements

# Package metadata
setup(
    name="tfm-mantenimiento-predictivo",
    version="1.0.0",
    author="Antonio",
    author_email="antonio@universidad.es",
    description="Sistema de Mantenimiento Predictivo Industrial - Ecosistema Completo TFM",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/antonio/tfm-mantenimiento-predictivo",
    project_urls={
        "Bug Tracker": "https://github.com/antonio/tfm-mantenimiento-predictivo/issues",
        "Documentation": "https://github.com/antonio/tfm-mantenimiento-predictivo/wiki",
        "Source Code": "https://github.com/antonio/tfm-mantenimiento-predictivo",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "dashboard": [
            "dash>=2.0.0",
            "plotly>=5.0.0",
            "dash-bootstrap-components>=1.0.0",
        ],
        "all": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0", 
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
            "dash>=2.0.0",
            "plotly>=5.0.0",
            "dash-bootstrap-components>=1.0.0",
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    entry_points={
        "console_scripts": [
            "tfm-pipeline=tfm_pipeline:main",
            "tfm-process-data=data_processor:main",
            "tfm-generate-orders=ot_generator:main",
            "tfm-learning=learning_system:main",
        ],
    },
    keywords=[
        "predictive maintenance", "industrial IoT", "anomaly detection", 
        "machine learning", "condition monitoring", "compressors",
        "isolation forest", "dbscan", "ensemble learning", "TFM"
    ],
    zip_safe=False,
)

# Post-installation setup
def post_install():
    """Ejecuta configuración post-instalación"""
    import os
    import json
    from pathlib import Path

    # Crear directorios necesarios
    directories = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "models",
        "reports/tfm_reproduction",
        "reports/anomalies",
        "reports/maintenance_orders",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

    # Verificar configuración
    config_path = "config/config.json"
    if os.path.exists(config_path):
        print(f"✅ Configuración encontrada: {config_path}")
    else:
        print(f"⚠️  Configuración no encontrada: {config_path}")
        print("   Ejecutar: python -c 'from src.tfm_pipeline import crear_configuracion_inicial; crear_configuracion_inicial()'")

    print("\n🎉 Instalación completada exitosamente!")
    print("📖 Consulta README.md para instrucciones de uso")

if __name__ == "__main__":
    # Ejecutar setup
    setup()

    # Post-instalación si se ejecuta directamente
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        post_install()
