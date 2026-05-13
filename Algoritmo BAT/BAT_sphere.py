import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Parámetros configurables
# -------------------------------
NUM_ITERACIONES = 80
NUM_BATS = 20
PAUSA = 0.25
RANGO = (-3.0, 3.0)
GRID_RES = 160
TRAIL_LENGTH = 6
SEED = 123

rng = np.random.default_rng(SEED)

# Parámetros BAT
FMIN = 0.0
FMAX = 2.0

ALPHA = 0.9        # disminución de loudness
GAMMA = 0.9        # aumento de pulse rate

A0 = 1.0           # loudness inicial
R0 = 0.1           # pulse rate inicial

# -------------------------------
# Función objetivo
# -------------------------------
def sphere(x, y):
    return -x**2 - y**2

# -------------------------------
# Inicialización
# -------------------------------
lo, hi = RANGO

# posiciones
bats = rng.uniform(lo, hi, size=(NUM_BATS, 2))

# velocidades
velocities = np.zeros((NUM_BATS, 2))

# frecuencias
frequencies = np.zeros(NUM_BATS)

# loudness y pulse rate
A = np.ones(NUM_BATS) * A0
R = np.ones(NUM_BATS) * R0

# evaluación inicial
values = np.array([sphere(x, y) for x, y in bats])

# mejor solución global
best_idx = np.argmax(values)
best_pos = bats[best_idx].copy()
best_val = values[best_idx]

# historial
history = [[bats[i].copy()] for i in range(NUM_BATS)]

# -------------------------------
# Malla para gráficas
# -------------------------------
X = np.linspace(lo, hi, GRID_RES)
Y = np.linspace(lo, hi, GRID_RES)
Xg, Yg = np.meshgrid(X, Y)
Zg = sphere(Xg, Yg)

plt.ion()

fig = plt.figure(figsize=(13,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

cont = ax1.contourf(Xg, Yg, Zg, 40, cmap="viridis")
cbar = plt.colorbar(cont, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Valor de la función")

# -------------------------------
# Función de dibujo
# -------------------------------
def dibujar(iteracion):

    ax1.clear()
    ax2.clear()

    ax1.contourf(Xg, Yg, Zg, 40, cmap="viridis")

    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)

    ax1.set_title(f"BAT - Contorno 2D (Iter {iteracion}/{NUM_ITERACIONES})")

    # rastros
    if TRAIL_LENGTH > 0:
        for i in range(NUM_BATS):
            trail = np.array(history[i][-TRAIL_LENGTH:])
            if trail.shape[0] > 1:
                ax1.plot(trail[:,0], trail[:,1],
                         linewidth=1,
                         alpha=0.6)

    # bats
    ax1.scatter(bats[:,0], bats[:,1],
                c='blue',
                s=35,
                label='bats')

    # mejor global
    ax1.scatter(best_pos[0], best_pos[1],
                c='red',
                s=160,
                marker='*',
                label='best')

    ax1.legend(loc='upper right')

    # superficie 3D
    ax2.plot_surface(Xg, Yg, Zg,
                     cmap="viridis",
                     alpha=0.85,
                     linewidth=0)

    current_vals = np.array([sphere(x,y) for x,y in bats])

    ax2.scatter(bats[:,0], bats[:,1],
                current_vals,
                c='blue',
                s=35)

    ax2.scatter(best_pos[0], best_pos[1],
                best_val,
                c='red',
                s=180,
                marker='*')

    ax2.set_title("BAT - Superficie 3D")

    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)

    ax2.view_init(elev=35, azim=-60)

    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)
    time.sleep(PAUSA)

# -------------------------------
# Bucle principal BAT
# -------------------------------
for t in range(1, NUM_ITERACIONES + 1):

    for i in range(NUM_BATS):

        # generar frecuencia
        beta = rng.random()

        frequencies[i] = FMIN + (FMAX - FMIN) * beta

        # actualizar velocidad
        velocities[i] = velocities[i] + \
                        (bats[i] - best_pos) * frequencies[i]

        # actualizar posición
        new_solution = bats[i] + velocities[i]

        # búsqueda local
        if rng.random() > R[i]:
            epsilon = rng.uniform(-1, 1, 2)
            new_solution = best_pos + epsilon * np.mean(A)

        # limitar rango
        new_solution = np.clip(new_solution, lo, hi)

        # evaluar
        new_value = sphere(new_solution[0], new_solution[1])

        # aceptar solución
        if (new_value > values[i]) and (rng.random() < A[i]):

            bats[i] = new_solution
            values[i] = new_value

            # actualizar loudness
            A[i] = ALPHA * A[i]

            # actualizar pulse rate
            R[i] = R0 * (1 - np.exp(-GAMMA * t))

        # guardar historial
        history[i].append(bats[i].copy())

    # actualizar mejor global
    idx = np.argmax(values)

    if values[idx] > best_val:
        best_val = values[idx]
        best_pos = bats[idx].copy()

    # dibujar
    dibujar(t)

# -------------------------------
# Resultado final
# -------------------------------
print(f"\nResultado final BAT -> best = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f = {best_val:.6f}")

plt.ioff()
plt.show()