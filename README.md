# FFT em Programação Competitiva — Relatório Cálculo 4

Este repositório reúne os códigos Python usados no relatório de Cálculo 4
sobre Transformada Rápida de Fourier (FFT) aplicada à multiplicação de
polinômios e a um problema típico de programação competitiva
(Fuzzy Search — Codeforces 528D).

## Estrutura do repositório

- `poly_fft_functions.py`  
  Funções básicas para multiplicação de polinômios:
  - `naive_polynomial_multiply(a, b)`: implementação ingênua em O(n^2)
  - `fft_polynomial_multiply(a, b)`: implementação via FFT (NumPy), em O(n log n).

- `run_fft_benchmark.py`  
  Script que:
  - gera coeficientes aleatórios,
  - aplica os dois métodos,
  - mede os tempos de execução,
  - verifica se os resultados coincidem,
  - e gera a figura `comparison_time.png` usada no relatório.

- `fuzzy_search_fft.py`  
  Implementação em Python do problema **Fuzzy Search (Codeforces 528D)**
  usando correlação via FFT (NumPy). Mostra como a mesma ideia de
  convolução/FFT é usada em um problema real de programação competitiva.

## Requisitos

- Python 3.9+ (qualquer versão recente deve funcionar)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/) (apenas para o benchmark/grafico)

Instalação rápida (opcionalmente em um ambiente virtual):

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install numpy matplotlib
