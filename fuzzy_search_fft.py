"""
fuzzy_search_fft.py

Solucao para o problema Fuzzy Search (Codeforces 528D)
usando correlacao via FFT (NumPy).

Formato esperado de entrada (adaptado ao relatorio):
n m k
S
T

No Codeforces original, o formato e levemente diferente
(veja o enunciado oficial).
"""

import sys
import numpy as np


def fft_convolution(a, b):
    """
    Convolucao linear via FFT real (rfft/irfft) com arredondamento.

    Parametros
    ----------
    a, b : arrays de float (0.0 ou 1.0, no contexto do problema)

    Retorna
    -------
    np.ndarray (np.int64)
        Resultado da convolucao linear de a e b.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    m = len(b)

    size = 1
    while size < n + m - 1:
        size <<= 1

    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    fc = fa * fb
    c = np.fft.irfft(fc, size)[: n + m - 1]
    return np.rint(c).astype(np.int64)


def main():
    data = sys.stdin.buffer.read().split()
    if len(data) < 5:
        # entrada invalida / incompleta
        print(0)
        return

    n = int(data[0])
    m = int(data[1])
    k = int(data[2])
    S = data[3].decode().strip()
    T = data[4].decode().strip()

    assert len(S) == n and len(T) == m

    total = np.zeros(n, dtype=np.int64)
    alphabet = ["A", "C", "G", "T"]

    for ch_a in alphabet:
        # (1) Dilatacao: marcar posicoes de S cobertas por ch_a em raio k
        diff = np.zeros(n + 1, dtype=np.int32)
        for i, ch in enumerate(S):
            if ch == ch_a:
                L = max(0, i - k)
                R = min(n - 1, i + k)
                diff[L] += 1
                diff[R + 1] -= 1

        ok = (np.cumsum(diff[:-1]) > 0).astype(np.float64)  # 0/1

        # (2) Indicador do padrao para ch_a (invertido para correlacao)
        pat = np.fromiter(
            (1.0 if ch == ch_a else 0.0 for ch in T),
            dtype=np.float64,
            count=m,
        )[::-1]

        # (3) Correlacao via convolucao FFT
        conv = fft_convolution(ok, pat)  # len = n + m - 1

        # (4) Acumular coincidencias por deslocamento
        # conv[l + m - 1] = matches da letra ch_a no deslocamento l
        max_shift = n - m + 1
        total[:max_shift] += conv[m - 1 : m - 1 + max_shift]

    # (5) Deslocamentos validos: soma == m
    max_shift = n - m + 1
    ans = int(np.count_nonzero(total[:max_shift] == m))
    print(ans)


if __name__ == "__main__":
    main()
