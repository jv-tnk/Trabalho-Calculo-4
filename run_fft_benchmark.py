"""
run_fft_benchmark.py

Script para comparar o tempo de execucao entre:
- naive_polynomial_multiply (O(n^2))
- fft_polynomial_multiply   (O(n log n))

Gera a figura comparison_time.png com o grafico de desempenho.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from poly_fft_functions import naive_polynomial_multiply, fft_polynomial_multiply


def main():
    # tamanhos usados no relatorio
    sizes = [10, 50, 100, 150, 200]
    naive_times = []
    fft_times = []

    np.random.seed(42)

    for n in sizes:
        # coeficientes inteiros aleatorios em [0, 9]
        a = np.random.randint(0, 10, size=n)
        b = np.random.randint(0, 10, size=n)

        # metodo ingenuo
        t0 = time.perf_counter()
        res_naive = naive_polynomial_multiply(a, b)
        naive_times.append(time.perf_counter() - t0)

        # metodo via FFT
        t0 = time.perf_counter()
        res_fft = fft_polynomial_multiply(a, b)
        fft_times.append(time.perf_counter() - t0)

        # checagem de corretude
        assert np.array_equal(res_naive, res_fft), "Resultados diferentes entre ingênuo e FFT!"

    # imprime tabela para a secao de Resultados
    print("n, naive_time_s, fft_time_s, speedup(naive/fft)")
    for n, tn, tf in zip(sizes, naive_times, fft_times):
        speedup = tn / tf if tf > 0 else float("inf")
        print(f"{n}, {tn:.8e}, {tf:.8e}, {speedup:.2f}x")

    # plota grafico
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, naive_times, marker="o", label="Ingenuo (O(n^2))")
    plt.plot(sizes, fft_times, marker="s", label="FFT (O(n log n))")
    plt.xlabel("Tamanho do polinomio (n)")
    plt.ylabel("Tempo de execucao (s)")
    plt.title("Comparacao de tempo para multiplicacao de polinomios")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_time.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
