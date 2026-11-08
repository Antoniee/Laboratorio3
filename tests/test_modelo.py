import sys
from pathlib import Path
import pytest
import numpy as np

# ruta 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelo import (construir_matriz_A, generar_malla_senales, 
                    generar_ruido, simular_lecturas, reconstruir_senales, calcular_metricas)

# Grupo 1
def test_caso_manual_sin_ruido():
    c = 0.5
    A = construir_matriz_A(c)
    x_manual = np.array([[0.5], [-0.5]])
    ruido_cero = np.zeros_like(x_manual)
    
    b_calculado = simular_lecturas(A, x_manual, ruido_cero)
    # Calculado a mano: [0.5 + 0.5*(-0.5), 0.5*0.5 + (-0.5)] = [0.25, -0.25]
    b_esperado = np.array([[0.25], [-0.25]])
    
    assert np.allclose(b_calculado, b_esperado)
    x_hat = reconstruir_senales(A, b_calculado)
    assert np.allclose(x_hat, x_manual)

# Grupo 2
def test_contratos_y_singularidad():
    with pytest.raises(ValueError, match="matriz singular"):
        construir_matriz_A(1.0)
    with pytest.raises(ValueError, match="matriz singular"):
        construir_matriz_A(-1.0)
        
    A_sensible = construir_matriz_A(0.99)
    assert A_sensible.shape == (2, 2)

    A = construir_matriz_A(0.2)
    with pytest.raises(ValueError, match="Dimensiones incompatibles"):
        simular_lecturas(A, np.ones((3, 10)), np.zeros((3, 10)))
        
    x_con_nan = np.array([[1.0, np.nan], [0.5, 0.5]])
    with pytest.raises(ValueError, match="no finitos"):
        simular_lecturas(A, x_con_nan, np.zeros((2, 2)))

# Grupo 3
@pytest.mark.parametrize("c", [0.2, 0.99])
def test_identidades_algebraicas(c):
    A = construir_matriz_A(c)
    tol = 1e-12
    
    lambdas, V = np.linalg.eigh(A)
    assert np.allclose(A @ V, V @ np.diag(lambdas), atol=tol)
    
    Q, R = np.linalg.qr(A)
    assert np.allclose(Q @ R, A, atol=tol)
    assert np.allclose(Q.T @ Q, np.eye(2), atol=tol)
    
    U, S, Vt = np.linalg.svd(A)
    assert np.allclose(U @ np.diag(S) @ Vt, A, atol=tol)

# Grupo 4
def test_estadisticas_y_clasificacion_manual():
    # Señal 1: error = [0.05, 0.12] -> norma = sqrt(0.0025 + 0.0144) = 0.13 (> 0.1)
    # Señal 2: error = [0.03, 0.04] -> norma = sqrt(0.0009 + 0.0016) = 0.05 (<= 0.1)
    x_real = np.array([[0.0, 0.0], [0.0, 0.0]])
    x_hat = np.array([[0.05, 0.03], [0.12, 0.04]])
    
    A = construir_matriz_A(0.2)
    b = A @ x_hat
    
    metrics = calcular_metricas(x_real, x_hat, A, b)
    
    assert np.isclose(metrics["media_error"][0], 0.04)  # (0.05 + 0.03) / 2
    assert np.isclose(metrics["media_error"][1], 0.08)  # (0.12 + 0.04) / 2
    assert np.allclose(metrics["error_norma"], np.array([0.13, 0.05]))
    assert np.isclose(metrics["prop_umbral"], 0.5)      # 1 de 2 señales supera 0.1
    assert metrics["idx_max"] == 0                      # El máximo (0.13) está en el índice 0

# Grupo 5
def test_reproducibilidad_y_no_mutacion():
    ruido_1 = generar_ruido()
    ruido_2 = generar_ruido()
    assert np.array_equal(ruido_1, ruido_2)
    
    x_original = generar_malla_senales()
    x_clon = x_original.copy()
    
    A = construir_matriz_A(0.99)
    b = simular_lecturas(A, x_original, ruido_1)
    x_hat = reconstruir_senales(A, b)
    _ = calcular_metricas(x_original, x_hat, A, b)
    
    assert np.array_equal(x_original, x_clon)
