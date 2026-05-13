"""
algoritmos.py
=============
Los 10 algoritmos metaheurísticos del repositorio como funciones
N-dimensionales, sin visualización, listos para evaluación masiva.

Todos comparten la misma firma:
    algo(func, dim, bounds, num_iter, init_positions, **kwargs)
        → (best_pos, best_val, start_pos, elapsed_time)

Donde:
    func            : callable f(x: ndarray) → float  (maximización)
    dim             : int, número de dimensiones
    bounds          : (lo, hi) escalares
    num_iter        : int, iteraciones internas
    init_positions  : ndarray (n_pop, dim)  población inicial
    best_pos        : ndarray (dim,)
    best_val        : float
    start_pos       : ndarray (dim,) – posición inicial de referencia
    elapsed_time    : float (segundos)
"""

import numpy as np
import time


# ─── Utilidad interna ────────────────────────────────────────
def _to2d(arr: np.ndarray, dim: int) -> np.ndarray:
    """Garantiza array 2-D de forma (n, dim)."""
    arr = np.array(arr, dtype=float)
    return arr.reshape(-1, dim) if arr.ndim == 1 else arr


# ════════════════════════════════════════════════════════════
# 1. ABC – Artificial Bee Colony
# ════════════════════════════════════════════════════════════
def abc(func, dim, bounds, num_iter, init_positions, limit=15, **_):
    """
    Colonia de Abejas Artificiales (ABC).
    Referencia: Karaboga (2005).
    """
    lo, hi = bounds
    sources = _to2d(init_positions, dim).copy()
    n  = sources.shape[0]
    rng = np.random.default_rng()

    values = np.array([func(x) for x in sources])
    trials = np.zeros(n, dtype=int)
    bi = int(np.argmax(values))
    best_pos  = sources[bi].copy()
    best_val  = float(values[bi])
    start_pos = sources[0].copy()

    def _prob(vals):
        sh = vals - vals.min() + 1e-9
        return sh / sh.sum()

    t0 = time.time()
    for _ in range(num_iter):

        # ── Employed bees ──────────────────────────────────
        for i in range(n):
            # Elegir k ≠ i
            k = int(rng.integers(0, n - 1))
            if k >= i:
                k += 1
            k = k % n  # protección n==1
            phi = rng.uniform(-1.0, 1.0, size=dim)
            v   = np.clip(sources[i] + phi * (sources[i] - sources[k]), lo, hi)
            fv  = func(v)
            if fv > values[i]:
                sources[i], values[i], trials[i] = v, fv, 0
            else:
                trials[i] += 1

        bi = int(np.argmax(values))
        if values[bi] > best_val:
            best_val, best_pos = float(values[bi]), sources[bi].copy()

        # ── Onlooker bees ──────────────────────────────────
        probs = _prob(values)
        for _ in range(n):
            sel = int(rng.choice(n, p=probs))
            k   = int(rng.integers(0, n - 1))
            if k >= sel:
                k += 1
            k   = k % n
            phi = rng.uniform(-1.0, 1.0, size=dim)
            v   = np.clip(sources[sel] + phi * (sources[sel] - sources[k]), lo, hi)
            fv  = func(v)
            if fv > values[sel]:
                sources[sel], values[sel], trials[sel] = v, fv, 0
            else:
                trials[sel] += 1

        bi = int(np.argmax(values))
        if values[bi] > best_val:
            best_val, best_pos = float(values[bi]), sources[bi].copy()

        # ── Scout bees ─────────────────────────────────────
        for i in range(n):
            if trials[i] >= limit:
                sources[i] = rng.uniform(lo, hi, size=dim)
                values[i]  = func(sources[i])
                trials[i]  = 0

        bi = int(np.argmax(values))
        if values[bi] > best_val:
            best_val, best_pos = float(values[bi]), sources[bi].copy()

    return best_pos, best_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 2. ACO – Ant Colony Optimization continuo (ACOR)
# ════════════════════════════════════════════════════════════
def aco(func, dim, bounds, num_iter, init_positions,
        num_ants=10, q=0.5, xi=0.85, **_):
    """
    Optimización por Colonia de Hormigas para espacios continuos (ACOR).
    Referencia: Socha & Dorigo (2008).
    """
    lo, hi = bounds
    archive     = _to2d(init_positions, dim).copy()
    m           = archive.shape[0]
    rng         = np.random.default_rng()

    archive_vals = np.array([func(x) for x in archive])
    order        = np.argsort(-archive_vals)
    archive      = archive[order]
    archive_vals = archive_vals[order]

    best_pos  = archive[0].copy()
    best_val  = float(archive_vals[0])
    start_pos = archive[0].copy()

    # Pesos Gaussianos (rank-based)
    k_idx = np.arange(1, m + 1, dtype=float)
    raw_w = np.exp(-(k_idx - 1.0)**2 / (2.0 * (q * m)**2)) / (q * m * np.sqrt(2.0 * np.pi))
    w     = raw_w / raw_w.sum()
    cum_w = np.cumsum(w)

    t0 = time.time()
    for _ in range(num_iter):
        # Desviación estándar por solución y dimensión
        sigmas = np.zeros((m, dim))
        for i in range(m):
            diffs      = np.abs(archive[i] - archive)
            sigmas[i]  = np.maximum(xi * diffs.sum(axis=0) / max(1, m - 1), 1e-6)

        # Generar nuevas soluciones (hormigas)
        new_sols = np.zeros((num_ants, dim))
        new_vals = np.zeros(num_ants)
        for a in range(num_ants):
            r = rng.random()
            k = min(int(np.searchsorted(cum_w, r, side='right')), m - 1)
            sample       = np.clip(rng.normal(archive[k], sigmas[k]), lo, hi)
            new_sols[a]  = sample
            new_vals[a]  = func(sample)

        # Actualizar archivo (mantener las m mejores)
        combined      = np.vstack((archive, new_sols))
        combined_vals = np.concatenate((archive_vals, new_vals))
        order         = np.argsort(-combined_vals)
        archive       = combined[order][:m]
        archive_vals  = combined_vals[order][:m]

        if archive_vals[0] > best_val:
            best_val  = float(archive_vals[0])
            best_pos  = archive[0].copy()

    return best_pos, best_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 3. BAT – Bat Algorithm
# ════════════════════════════════════════════════════════════
def bat(func, dim, bounds, num_iter, init_positions,
        fmin=0.0, fmax=2.0, alpha=0.9, gamma=0.9, a0=1.0, r0=0.1, **_):
    """
    Algoritmo de Murciélagos (BAT).
    Referencia: Yang (2010).
    """
    lo, hi = bounds
    bats = _to2d(init_positions, dim).copy()
    n    = bats.shape[0]
    rng  = np.random.default_rng()

    velocities = np.zeros((n, dim))
    A          = np.ones(n) * a0
    R          = np.ones(n) * r0
    values     = np.array([func(x) for x in bats])

    bi = int(np.argmax(values))
    best_pos  = bats[bi].copy()
    best_val  = float(values[bi])
    start_pos = bats[0].copy()

    t0 = time.time()
    for t in range(1, num_iter + 1):
        for i in range(n):
            freq            = fmin + (fmax - fmin) * rng.random()
            velocities[i]  += (bats[i] - best_pos) * freq
            new_sol         = bats[i] + velocities[i]

            # Búsqueda local alrededor del mejor global
            if rng.random() > R[i]:
                eps     = rng.uniform(-1.0, 1.0, dim)
                new_sol = best_pos + eps * float(np.mean(A))

            new_sol = np.clip(new_sol, lo, hi)
            new_val = func(new_sol)

            if (new_val > values[i]) and (rng.random() < A[i]):
                bats[i]   = new_sol
                values[i] = new_val
                A[i]     *= alpha
                R[i]      = r0 * (1.0 - np.exp(-gamma * t))

        bi = int(np.argmax(values))
        if values[bi] > best_val:
            best_val, best_pos = float(values[bi]), bats[bi].copy()

    return best_pos, best_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 4. FA – Firefly Algorithm
# ════════════════════════════════════════════════════════════
def fa(func, dim, bounds, num_iter, init_positions,
       beta0=1.0, gamma=1.0, alpha=0.2, alpha_decay=0.98, **_):
    """
    Algoritmo de las Luciérnagas (FA).
    Referencia: Yang (2008).
    """
    lo, hi  = bounds
    X       = _to2d(init_positions, dim).copy()
    n       = X.shape[0]
    rng     = np.random.default_rng()
    scale   = float(hi - lo)   # escala de ruido

    vals    = np.array([func(x) for x in X])
    bi      = int(np.argmax(vals))
    best_pos  = X[bi].copy()
    best_val  = float(vals[bi])
    start_pos = X[0].copy()

    alpha_curr = alpha
    t0 = time.time()
    for _ in range(num_iter):
        # Ordenar de menor a mayor brillo (menor valor)
        order = np.argsort(vals)
        for idx_i in range(n):
            i  = order[idx_i]
            xi = X[i].copy()
            # Moverse hacia luciérnagas más brillantes
            for idx_j in range(idx_i):
                j    = order[idx_j]
                r    = np.linalg.norm(xi - X[j])
                beta = beta0 * np.exp(-gamma * r**2)
                noise = rng.normal(0.0, 1.0, dim)
                xi = np.clip(xi + beta * (X[j] - xi)
                             + alpha_curr * noise * scale * 0.1, lo, hi)
            X[i]    = xi
            vals[i] = func(xi)

        alpha_curr *= alpha_decay
        bi = int(np.argmax(vals))
        if vals[bi] > best_val:
            best_val, best_pos = float(vals[bi]), X[bi].copy()

    return best_pos, best_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 5. GA – Genetic Algorithm
# ════════════════════════════════════════════════════════════
def ga(func, dim, bounds, num_iter, init_positions,
       prob_cross=0.8, prob_mut=0.2, mut_range=0.3, **_):
    """
    Algoritmo Genético con cruce BLX-α y mutación uniforme.
    """
    lo, hi = bounds
    pop    = _to2d(init_positions, dim).copy()
    n      = pop.shape[0]
    rng    = np.random.default_rng()

    fitness = np.array([func(x) for x in pop])
    bi      = int(np.argmax(fitness))
    best_pos  = pop[bi].copy()
    best_val  = float(fitness[bi])
    start_pos = pop[0].copy()

    t0 = time.time()
    for _ in range(num_iter):
        # Selección por ruleta
        fp    = fitness - fitness.min() + 1e-9
        probs = fp / fp.sum()
        idx   = rng.choice(n, size=n, p=probs)
        parents = pop[idx]

        # Cruce aritmético (BLX-α)
        offspring = []
        for i in range(0, n, 2):
            p1, p2 = parents[i], parents[(i + 1) % n]
            if rng.random() < prob_cross:
                a = rng.random()
                offspring.extend([a * p1 + (1 - a) * p2,
                                   a * p2 + (1 - a) * p1])
            else:
                offspring.extend([p1.copy(), p2.copy()])
        pop = np.array(offspring[:n])

        # Mutación uniforme
        for ind in pop:
            if rng.random() < prob_mut:
                ind += rng.uniform(-mut_range, mut_range, dim)
                np.clip(ind, lo, hi, out=ind)

        fitness = np.array([func(x) for x in pop])
        bi      = int(np.argmax(fitness))
        if fitness[bi] > best_val:
            best_val, best_pos = float(fitness[bi]), pop[bi].copy()

    return best_pos, best_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 6. HS – Harmony Search
# ════════════════════════════════════════════════════════════
def hs(func, dim, bounds, num_iter, init_positions,
       hmcr=0.95, par=0.3, bw=0.1, **_):
    """
    Búsqueda Armónica (Harmony Search).
    Referencia: Geem et al. (2001).
    """
    lo, hi = bounds
    HM     = _to2d(init_positions, dim).copy()
    hms    = HM.shape[0]
    rng    = np.random.default_rng()

    HM_vals = np.array([func(x) for x in HM])
    bi      = int(np.argmax(HM_vals))
    best_pos  = HM[bi].copy()
    best_val  = float(HM_vals[bi])
    start_pos = HM[0].copy()

    bw_abs = bw * (hi - lo)   # bandwidth en unidades del dominio

    t0 = time.time()
    for _ in range(num_iter):
        new = np.empty(dim)
        for d in range(dim):
            if rng.random() < hmcr:
                val = HM[int(rng.integers(0, hms)), d]
                if rng.random() < par:
                    val += rng.uniform(-bw_abs, bw_abs)
            else:
                val = rng.uniform(lo, hi)
            new[d] = np.clip(val, lo, hi)

        new_val = func(new)
        worst   = int(np.argmin(HM_vals))
        if new_val > HM_vals[worst]:
            HM[worst]      = new
            HM_vals[worst] = new_val
            if new_val > best_val:
                best_val, best_pos = float(new_val), new.copy()

    return best_pos, best_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 7. PSO – Particle Swarm Optimization
# ════════════════════════════════════════════════════════════
def pso(func, dim, bounds, num_iter, init_positions,
        w_max=0.9, w_min=0.4, c1=1.5, c2=1.5, v_ratio=0.2, **_):
    """
    Optimización por Enjambre de Partículas (PSO) con inercia variable.
    Referencia: Kennedy & Eberhart (1995).
    """
    lo, hi = bounds
    pos    = _to2d(init_positions, dim).copy()
    n      = pos.shape[0]
    rng    = np.random.default_rng()
    vmax   = v_ratio * (hi - lo)

    vel       = rng.uniform(-vmax, vmax, size=(n, dim))
    pbest_pos = pos.copy()
    pbest_val = np.array([func(x) for x in pbest_pos])

    gi = int(np.argmax(pbest_val))
    gbest_pos = pbest_pos[gi].copy()
    gbest_val = float(pbest_val[gi])
    start_pos = pos[0].copy()

    t0 = time.time()
    for t in range(1, num_iter + 1):
        w   = w_max - (w_max - w_min) * (t - 1) / max(1, num_iter - 1)
        r1  = rng.random((n, dim))
        r2  = rng.random((n, dim))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, lo, hi)

        fitness = np.array([func(x) for x in pos])
        mask    = fitness > pbest_val
        pbest_pos[mask] = pos[mask]
        pbest_val[mask] = fitness[mask]

        gi = int(np.argmax(pbest_val))
        if pbest_val[gi] > gbest_val:
            gbest_val, gbest_pos = float(pbest_val[gi]), pbest_pos[gi].copy()

    return gbest_pos, gbest_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 8. Hill Climbing  (Climbing)
# ════════════════════════════════════════════════════════════
def hill_climbing(func, dim, bounds, num_iter, init_positions,
                  step_size=0.5, step_decay=0.5, min_step=0.001, **_):
    """
    Ascenso de Colinas (Hill Climbing) con decaimiento de paso.
    Direcciones cardinales: ±eᵢ para cada dimensión i.
    """
    lo, hi = bounds
    init   = _to2d(init_positions, dim)

    # Partir del mejor punto de la población inicial
    init_vals = np.array([func(x) for x in init])
    x         = init[int(np.argmax(init_vals))].copy()
    fx        = float(func(x))
    start_pos = x.copy()

    # Direcciones cardinales en N dimensiones: 2·dim vectores
    dirs = np.vstack([np.eye(dim), -np.eye(dim)])

    step = step_size
    t0   = time.time()
    for _ in range(num_iter):
        if step < min_step:
            break
        neighbors = np.clip(x + dirs * step, lo, hi)
        f_neigh   = np.array([func(v) for v in neighbors])
        best_n    = int(np.argmax(f_neigh))
        if f_neigh[best_n] > fx:
            x, fx = neighbors[best_n], float(f_neigh[best_n])
        else:
            step *= step_decay

    return x, fx, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 9. Búsqueda Aleatoria  (Random Search)
# ════════════════════════════════════════════════════════════
def random_search(func, dim, bounds, num_iter, init_positions, **_):
    """
    Búsqueda Aleatoria pura: muestrea puntos al azar en cada iteración
    y conserva el mejor encontrado.
    """
    lo, hi = bounds
    init   = _to2d(init_positions, dim)
    rng    = np.random.default_rng()

    # Evaluar población inicial
    init_vals = np.array([func(x) for x in init])
    bi        = int(np.argmax(init_vals))
    best_pos  = init[bi].copy()
    best_val  = float(init_vals[bi])
    start_pos = best_pos.copy()

    t0 = time.time()
    for _ in range(num_iter):
        x  = rng.uniform(lo, hi, dim)
        fv = func(x)
        if fv > best_val:
            best_val, best_pos = float(fv), x.copy()

    return best_pos, best_val, start_pos, time.time() - t0


# ════════════════════════════════════════════════════════════
# 10. Búsqueda Local  (Local Search estocástica con vecindario ampliado)
# ════════════════════════════════════════════════════════════
def local_search(func, dim, bounds, num_iter, init_positions,
                 step_size=0.5, step_decay=0.5, min_step=0.001, **_):
    """
    Búsqueda Local con vecindario ampliado:
    Incluye direcciones cardinales + perturbaciones diagonales aleatorias.
    Para N grandes usa 3·dim direcciones adicionales (evita explosión combinatoria).
    """
    lo, hi = bounds
    init   = _to2d(init_positions, dim)
    rng    = np.random.default_rng()

    init_vals = np.array([func(x) for x in init])
    x         = init[int(np.argmax(init_vals))].copy()
    fx        = float(func(x))
    start_pos = x.copy()

    # Direcciones cardinales
    cardinal = np.vstack([np.eye(dim), -np.eye(dim)])
    # Direcciones diagonales aleatorias (max 3·dim adicionales)
    n_diag = min(3 * dim, 30)
    diag   = rng.choice([-1.0, 0.5, 0.0, -0.5, 1.0],
                         size=(n_diag, dim))
    # Eliminar dirección nula
    mask = ~np.all(diag == 0.0, axis=1)
    diag = diag[mask]
    # Normalizar para que sean unit-like
    norms = np.linalg.norm(diag, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    diag  = diag / norms

    dirs = np.vstack([cardinal, diag])

    step = step_size
    t0   = time.time()
    for _ in range(num_iter):
        if step < min_step:
            break
        neighbors = np.clip(x + dirs * step, lo, hi)
        f_neigh   = np.array([func(v) for v in neighbors])
        best_n    = int(np.argmax(f_neigh))
        if f_neigh[best_n] > fx:
            x, fx = neighbors[best_n], float(f_neigh[best_n])
        else:
            step *= step_decay

    return x, fx, start_pos, time.time() - t0
