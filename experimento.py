import numpy as np
import matplotlib.pyplot as plt
from modelo import (construir_matriz_A, generar_malla_senales, 
                    generar_ruido, simular_lecturas, reconstruir_senales, calcular_metricas)

def ejecutar_diagnostico_matricial(c: float) -> dict:
    A = construir_matriz_A(c)
    
    # 1. Autovalores y Autovectores
    lambdas, V = np.linalg.eigh(A)
    disc_v = np.linalg.norm(A @ V - V @ np.diag(lambdas))
    
    # 2. Factorización QR
    Q, R = np.linalg.qr(A)
    disc_qr = np.linalg.norm(Q @ R - A)
    disc_q_ort = np.linalg.norm(Q.T @ Q - np.eye(2))
    
    # 3. SVD
    U, S, Vt = np.linalg.svd(A)
    disc_svd = np.linalg.norm(U @ np.diag(S) @ Vt - A)
    
    # Número de condición k_2
    k_2 = np.max(S) / np.min(S)
    
    return {
        "lambdas": lambdas, "S": S, "k_2": k_2,
        "disc_v": disc_v, "disc_qr": disc_qr, "disc_q_ort": disc_q_ort, "disc_svd": disc_svd
    }

def correr_experimento_completo():
    x_real = generar_malla_senales()
    ruido = generar_ruido()
    
    configs = [0.2, 0.99]
    resultados = {}
    
    for c in configs:
        A = construir_matriz_A(c)
        b = simular_lecturas(A, x_real, ruido)
        x_hat = reconstruir_senales(A, b)
        
        diag_mat = ejecutar_diagnostico_matricial(c)
        metrics = calcular_metricas(x_real, x_hat, A, b)
        
        resultados[c] = {"diag": diag_mat, "metrics": metrics}
    
    # --- Generación de la Figura de Dos Paneles ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    ticks = np.linspace(-1, 1, 21)
    
    vmax = max(np.max(resultados[0.2]["metrics"]["error_norma"]), 
               np.max(resultados[0.99]["metrics"]["error_norma"]))
    
    for i, c in enumerate(configs):
        err_malla = resultados[c]["metrics"]["error_norma"].reshape(21, 21)
        im = axes[i].pcolormesh(ticks, ticks, err_malla, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        axes[i].set_title(f"Norma del Error de Reconstrucción (c = {c})")
        axes[i].set_xlabel("Componente $x_1$")
        if i == 0: 
            axes[i].set_ylabel("Componente $x_2$")
        fig.colorbar(im, ax=axes[i], label="Error Absoluto (Norma L2)")
        
    plt.tight_layout()
    plt.savefig("mapa_errores.png", dpi=300)
    plt.close()
    print("\n[INFO] Gráfica guardada como 'mapa_errores.png'\n")
    
    # --- Impresión de Tablas Formateadas ---
    print("### TABLA 1: DIAGNÓSTICO MATRICIAL")
    print("|   c  | Autovalores (λ) | Valores Singulares (σ) | Condición κ₂ | Disc. QR | Disc. SVD |")
    print("|------|-----------------|-----------------------|--------------|----------|-----------|")
    for c in configs:
        d = resultados[c]["diag"]
        print(f"| {c:<4} | {str(np.round(d['lambdas'],4)):<15} | {str(np.round(d['S'],4)):<21} | {d['k_2']:<12.2f} | {d['disc_qr']:<8.2e} | {d['disc_svd']:<9.2e} |")
        
    print("\n### TABLA 2: RECONSTRUCCIÓN Y ESTADÍSTICAS")
    print("|   c  | Media Error (Comp 1, 2) | Desv. Est. Error | Error Máximo | Prop > 0.1 | Máx Residuo |")
    print("|------|-------------------------|------------------|--------------|------------|-------------|")
    for c in configs:
        m = resultados[c]["metrics"]
        media_str = str(np.round(m['media_error'], 5))
        std_str = str(np.round(m['std_error'], 5))
        print(f"| {c:<4} | {media_str:<23} | {std_str:<16} | {np.max(m['error_norma']):<12.4f} | {m['prop_umbral']:<10.2%} | {np.max(m['residuo_norma']):<11.4f} |")

if __name__ == "__main__":
    correr_experimento_completo()
