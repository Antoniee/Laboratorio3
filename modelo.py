import numpy as np

def construir_matriz_A(c: float) -> np.ndarray:
    """ con este código Construyo la matriz del sistema A(c) y valida singularidad."""
    if np.isclose(abs(c), 1.0):
        raise ValueError(f"El parámetro c={c} produce una matriz singular.")
    return np.array([[1.0, float(c)], 
                     [float(c), 1.0]], dtype=float)

def generar_malla_senales() -> np.ndarray:
    """
    Aquí genero la  malla de 21x21 combinaciones entre -1 y 1.
    Luego retorna un arreglo de forma (2, 441).
    En el Eje 0: Las 2 componentes reales (x1, x2).
    En el Eje 1: Las 441 señales independientes.
    """
    ticks = np.linspace(-1, 1, 21)
    X1, X2 = np.meshgrid(ticks, ticks)
    return np.vstack([X1.flatten(), X2.flatten()])

def generar_ruido(num_senales: int = 441) -> np.ndarray:
    """ Aquí genero una única realización de ruido normal con default_rng(7)."""
    rng = np.random.default_rng(7)
    return rng.normal(loc=0.0, scale=0.01, size=(2, num_senales))



def simular_lecturas(A: np.ndarray, x: np.ndarray, ruido: np.ndarray) -> np.ndarray:
    """Calcula b = A @ x + ruido usando producto matricial y adición vectorizada."""
    if A.shape != (2, 2) or x.ndim != 2 or x.shape[0] != 2 or ruido.shape != x.shape:
        raise ValueError("Dimensiones incompatibles para simulación.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(x)) and np.all(np.isfinite(ruido))):
        raise ValueError("Los arreglos contienen valores no finitos (NaN o Inf).")
    return (A @ x) + ruido


def reconstruir_senales(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Esta función recupera x_hat resolviendo múltiples columnas simultáneamente con solve."""
    return np.linalg.solve(A, b)

def calcular_metricas(x_real: np.ndarray, x_hat: np.ndarray, A: np.ndarray, b: np.ndarray) -> dict:
    """Esta función calcula errores, residuos y estadísticas reduciendo sobre el Eje 1."""
    error = x_hat - x_real
    residuo = (A @ x_hat) - b
    
    error_norma = np.linalg.norm(error, axis=0)
    residuo_norma = np.linalg.norm(residuo, axis=0)
    
    media_error = np.mean(error, axis=1, keepdims=True)
    std_error = np.std(error, axis=1, ddof=0, keepdims=True)
    
    proporcion_sobre_umbral = np.mean(error_norma > 0.1)
    idx_max_error = np.argmax(error_norma)
    
    return {
        "error_norma": error_norma,
        "residuo_norma": residuo_norma,
        "media_error": media_error.flatten(),
        "std_error": std_error.flatten(),
        "prop_umbral": proporcion_sobre_umbral,
        "idx_max": idx_max_error
    }
