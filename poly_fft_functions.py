"""
poly_fft_functions.py

Funcoes basicas para multiplicacao de polinomios:
- naive_polynomial_multiply: O(n^2)
- fft_polynomial_multiply: O(n log n) via FFT (NumPy)
"""

import numpy as np


def naive_polynomial_multiply(a, b):
    """
    Multiplica dois polinomios em O(n^2) usando duplo laco.

    Parametros
    ----------
    a, b : sequencias de numeros (lista, tuple, array)
        Coeficientes dos polinomios A(x) e B(x).

    Retorna
    -------
    np.ndarray
        Vetor de coeficientes de C(x) = A(x) * B(x).
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n, m = len(a), len(b)
    result = np.zeros(n + m - 1, dtype=np.int64)

    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]

    return result


def fft_polynomial_multiply(a, b):
    """
    Multiplica dois polinomios via convolucao linear usando FFT real.

    Usa rfft/irfft com zero padding ate a menor potencia de 2 >= n + m - 1.

    Parametros
    ----------
    a, b : sequencias de numeros (lista, tuple, array)
        Coeficientes dos polinomios A(x) e B(x).

    Retorna
    -------
    np.ndarray
        Vetor de coeficientes inteiros de C(x) = A(x) * B(x).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    m = len(b)
    size = 1
    # menor potencia de 2 >= n + m - 1
    while size < n + m - 1:
        size <<= 1

    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    fc = fa * fb
    c = np.fft.irfft(fc, size)[: n + m - 1]

    # arredonda para inteiros (esperado quando os coeficientes iniciais sao inteiros)
    return np.rint(c).astype(np.int64)


if __name__ == "__main__":
    # pequeno teste rapido
    A = [1, 2, 3]
    B = [4, 0, -1]

    c1 = naive_polynomial_multiply(A, B)
    c2 = fft_polynomial_multiply(A, B)

    print("A:", A)
    print("B:", B)
    print("naive:", c1)
    print("fft  :", c2)
    assert np.array_equal(c1, c2)
    print("OK: resultados coincidem.")
