import numpy as np
import matplotlib.pyplot as plt

def latin_hypercube(n_samples, dim, rng):
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(size=(n_samples, dim))
    samples = np.zeros((n_samples, dim))
    for j in range(dim):
        idx = rng.permutation(n_samples)
        samples[:, j] = cut[idx] + u[:, j] * (1.0 / n_samples)
    return np.clip(samples,0.0,1.0)


def maximin_selection(candidates, m, rng = None):
    N = candidates.shape[0]
    assert m <= N, "m must be less than or equal to the number of candidates"
    if rng is None:
        rng = np.random.default_rng()
    selected_idx = []
    
    first = rng.integers(0, N)
    selected_idx.append(first)
    
    while len(selected_idx) < m:
        sel = np.array(selected_idx)
        chosen = -1
        best_min_dist = -1.0
        for i in range(N):
            if i in sel:
                continue
            dist = np.linalg.norm(candidates[i] - candidates[sel], axis=1)
            min_dist = dist.min()
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                chosen = i
        selected_idx.append(chosen)
    return candidates[selected_idx]


rng = np.random.default_rng(2025)
numPuntosDeseados = 50
dim = 2
pool_size = 500

candidatos = latin_hypercube(pool_size, dim, rng)

seleccion_lhs = candidatos[:numPuntosDeseados, :]

seleccion_maximin = maximin_selection(candidatos, numPuntosDeseados, rng=rng)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(candidatos[:, 0], candidatos[:, 1], s=8, alpha=0.35, label='Candidatos LHS')
plt.scatter(seleccion_lhs[:, 0], seleccion_lhs[:, 1], s=50, c='r', marker='o',edgecolor='k', label='Seleccion LHS')
plt.title('Selección directo LHH')
plt.xlim([0,1]); plt.ylim([0,1]); plt.grid(True);plt.legend()


plt.subplot(1, 2, 2)
plt.scatter(candidatos[:, 0], candidatos[:, 1], s=8, alpha=0.35, label='Candidatos LHS')
plt.scatter(seleccion_maximin[:, 0], seleccion_maximin[:, 1], s=50, c='g', marker='o', edgecolor='k', label='Seleccion Maximin')
plt.title('Selección Maxinim')
plt.xlim([0,1]); plt.ylim([0,1]); plt.grid(True);plt.legend()

plt.tight_layout()
plt.show()