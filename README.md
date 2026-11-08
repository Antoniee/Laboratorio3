# Laboratorio 3 — Reconstrucción de señales y sensibilidad con NumPy

Este repositorio contiene la implementación del modelo matricial de sensores, el análisis de condicionamiento y sensibilidad al ruido, y las pruebas automatizadas con `pytest`.

## Ubicación de archivos y resultados
* `modelo.py`: Funciones matemáticas puras, validación de contratos y métricas.
* `experimento.py`: Orquestación de las simulaciones (`c = 0.2` y `c = 0.99`), diagnóstico matricial y exportación de resultados.
* `tests/test_modelo.py`: Suite de pruebas automatizadas (contine 5 grupos de validación).
* `requirements.txt`: Dependencias exactas del entorno entre otros elementos de interés(`numpy`, `matplotlib`, `pytest`).
* `INFORME.md`: Documento principal con tablas de resultados, análisis, respuestas y nota individual.
* `mapa_errores.png`: Figura comparativa generada al ejecutar el experimento.

## Parámetros de reproducibilidad
* **Malla de señales:** `21 × 21` puntos equiespaciados en `[-1, 1]`.
* **Generador de ruido:** `np.random.default_rng(7)` (distribución normal, media `0.0`, desviación estándar `0.01`).

## Comandos de instalación y ejecución

En tu terminal de  Ubuntu, puedes ejecutar:

```bash
# 1. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar las pruebas automatizadas
python -m pytest -v

# 4. Ejecutar el experimento y regenerar la figura
python experimento.py
```
