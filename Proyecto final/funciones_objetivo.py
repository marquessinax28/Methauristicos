"""
funciones_objetivo.py
  ackley    – función Ackley generalizada a N dims
  peaks     – función Peaks (usa solo las 2 primeras dims)
  sphere    – función Sphere negada (maximizar = minimizar x²)
  rastrigin – función Rastrigin (maximización)
"""

import numpy as np


def ackley(x):
    """
    Función Ackley N-dimensional (maximización).
    Para 2D es idéntica a la versión del repositorio original.
    Rango típico: [-3, 3]^n  |  Máximo: esquinas del dominio
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    A, B, C = 20.0, 0.2, 2.0 * np.pi
    term1 = -A * np.exp(-B * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(C * x)) / n)
    return term1 + term2 + A + np.exp(1.0)


def peaks(x):
    """
    Función Peaks (solo usa las 2 primeras dimensiones del vector x).
    Para dims > 2 las dimensiones restantes no afectan el valor.
    Rango típico: [-3, 3]^n  |  Máximo global ≈ 8.106
    """
    x = np.asarray(x, dtype=float)
    a, b = x[0], x[1]
    return (3.0 * (1.0 - a)**2 * np.exp(-(a**2) - (b + 1.0)**2)
            - 10.0 * (a / 5.0 - a**3 - b**5) * np.exp(-a**2 - b**2)
            - (1.0 / 3.0) * np.exp(-(a + 1.0)**2 - b**2))


def sphere(x):
    """
    Función Sphere N-dimensional negada para maximización.
    sphere(x) = -Σ xᵢ²   →  máximo (= 0) en el origen.
    Rango típico: [-3, 3]^n
    """
    x = np.asarray(x, dtype=float)
    return -np.sum(x**2)


def rastrigin(x):
    """
    Función Rastrigin N-dimensional (maximización, igual que el repo original).
    f(x) = n·10 + Σ(xᵢ² – 10·cos(2π·xᵢ))
    Rango típico: [-3, 3]^n
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    return n * 10.0 + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))
