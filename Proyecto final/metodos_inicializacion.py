"""
metodos_inicializacion.py
=========================
Métodos de inicialización de poblaciones para N dimensiones.

Funciones disponibles:
  inicializacion_lhs   – Latin Hypercube Sampling
  maximin_distance     – Selección maximin desde un pool LHS
"""

import numpy as np


def inicializacion_lhs(n_samples: int, dim: int, lo: float, hi: float,
                       rng=None) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHS).

    Genera `n_samples` puntos en [lo, hi]^dim con buena cobertura
    del espacio mediante permutaciones por dimensión.

    Parámetros
    ----------
    n_samples : número de muestras
    dim       : número de dimensiones
    lo, hi    : límites del dominio (escalares)
    rng       : numpy.random.Generator (opcional, para reproducibilidad)

    Retorna
    -------
    array de forma (n_samples, dim) con valores en [lo, hi]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Malla [0,1] con n_samples intervalos de ancho 1/n_samples
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    samples = np.zeros((n_samples, dim))

    for j in range(dim):
        idx = rng.permutation(n_samples)          # permutación aleatoria
        u   = rng.uniform(size=n_samples)          # ruido dentro del intervalo
        samples[:, j] = cut[idx] + u / n_samples  # punto dentro del sub-intervalo

    samples = np.clip(samples, 0.0, 1.0)
    # Escalar de [0,1] a [lo, hi]
    return samples * (hi - lo) + lo


def maximin_distance(n_samples: int, dim: int, lo: float, hi: float,
                     pool_size: int = None, rng=None) -> np.ndarray:
    """
    Selección Maximin desde un pool LHS grande.

    Genera un pool grande de candidatos LHS y selecciona de forma
    greedy los `n_samples` puntos que maximizan la distancia mínima
    entre pares (estrategia maximin).

    Parámetros
    ----------
    n_samples : número de puntos a seleccionar
    dim       : número de dimensiones
    lo, hi    : límites del dominio (escalares)
    pool_size : tamaño del pool de candidatos (default = max(150, n_samples·8))
    rng       : numpy.random.Generator (opcional)

    Retorna
    -------
    array de forma (n_samples, dim)
    """
    if rng is None:
        rng = np.random.default_rng()
    if pool_size is None:
        pool_size = max(150, n_samples * 8)

    # Generar pool de candidatos con LHS
    candidates = inicializacion_lhs(pool_size, dim, lo, hi, rng)

    # Si se piden más puntos que el pool, devolver directamente
    if n_samples >= pool_size:
        return candidates[:n_samples]

    # Selección greedy maximin
    selected_idx = []
    selected_set = set()

    first = int(rng.integers(0, pool_size))
    selected_idx.append(first)
    selected_set.add(first)

    while len(selected_idx) < n_samples:
        sel_pts = candidates[selected_idx]   # ya seleccionados
        best_min_dist = -1.0
        chosen = -1

        for i in range(pool_size):
            if i in selected_set:
                continue
            # Distancia euclidiana al punto más cercano ya seleccionado
            dists  = np.linalg.norm(candidates[i] - sel_pts, axis=1)
            min_d  = dists.min()
            if min_d > best_min_dist:
                best_min_dist = min_d
                chosen = i

        selected_idx.append(chosen)
        selected_set.add(chosen)

    return candidates[selected_idx]
