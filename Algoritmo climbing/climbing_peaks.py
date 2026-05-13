import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Parámetros
# -------------------------------
NUM_ITERACIONES = 80
PAUSA = 0.25

RANGO = (-3.0, 3.0)

GRID_RES = 160

TRAIL_LENGTH = 20

SEED = 123

rng = np.random.default_rng(SEED)

# Hill Climbing
STEP_SIZE = 0.5
STEP_DECAY = 0.5
MIN_STEP = 0.001

# -------------------------------
# Función objetivo
# -------------------------------
def peaks(x, y):
    return 3*(1-x)**2*np.exp(-(x**2) - (y+1)**2) \
           - 10*(x/5 - x**3 - y**5)*np.exp(-x**2 - y**2) \
           - 1/3*np.exp(-(x+1)**2 - y**2) 

# -------------------------------
# Inicialización
# -------------------------------
lo, hi = RANGO

# punto inicial aleatorio
x = rng.uniform(lo, hi, size=2)

fx = peaks(x[0], x[1])

best_pos = x.copy()
best_val = fx

# historial
history = [x.copy()]

# -------------------------------
# Direcciones vecinas
# -------------------------------
dirs = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
])

# -------------------------------
# Malla para gráficas
# -------------------------------
X = np.linspace(lo, hi, GRID_RES)
Y = np.linspace(lo, hi, GRID_RES)

Xg, Yg = np.meshgrid(X, Y)

Zg = peaks(Xg, Yg)

# -------------------------------
# Configuración gráfica
# -------------------------------
plt.ion()

fig = plt.figure(figsize=(13,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122, projection='3d')

cont = ax1.contourf(Xg, Yg, Zg, 40, cmap="viridis")

cbar = plt.colorbar(cont, ax=ax1,
                    fraction=0.046,
                    pad=0.04)

cbar.set_label("Valor de la función")

# -------------------------------
# Función de dibujo
# -------------------------------
def dibujar(iteracion):

    ax1.clear()
    ax2.clear()

    # Contorno 2D
    ax1.contourf(Xg, Yg, Zg, 40, cmap="viridis")

    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)

    ax1.set_title(
        f"Hill Climbing - Contorno 2D "
        f"(Iter {iteracion}/{NUM_ITERACIONES})"
    )

    # Rastro
    if len(history) > 1:

        trail = np.array(history[-TRAIL_LENGTH:])

        ax1.plot(
            trail[:,0],
            trail[:,1],
            linewidth=2,
            alpha=0.7
        )

    # Punto actual
    ax1.scatter(
        x[0],
        x[1],
        c='blue',
        s=60,
        label='posición actual'
    )

    # Mejor punto
    ax1.scatter(
        best_pos[0],
        best_pos[1],
        c='red',
        s=180,
        marker='*',
        label='mejor solución'
    )

    ax1.legend(loc='upper right')

    # ---------------------------
    # Superficie 3D
    # ---------------------------
    ax2.plot_surface(
        Xg,
        Yg,
        Zg,
        cmap="viridis",
        alpha=0.85,
        linewidth=0
    )

    ax2.scatter(
        x[0],
        x[1],
        fx,
        c='blue',
        s=60
    )

    ax2.scatter(
        best_pos[0],
        best_pos[1],
        best_val,
        c='red',
        s=180,
        marker='*'
    )

    ax2.set_title("Hill Climbing - Superficie 3D")

    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)

    ax2.view_init(elev=35, azim=-60)

    plt.tight_layout()

    plt.draw()

    plt.pause(PAUSA)

    time.sleep(PAUSA)

# -------------------------------
# Algoritmo Hill Climbing
# -------------------------------
step_size = STEP_SIZE

for t in range(1, NUM_ITERACIONES + 1):

    if step_size <= MIN_STEP:
        break

    # generar vecinos
    vecinos = np.clip(
        x + dirs * step_size,
        lo,
        hi
    )

    # evaluar vecinos
    f_vecinos = np.array([
        peaks(v[0], v[1])
        for v in vecinos
    ])

    # mejor vecino
    idx_mejor = np.argmax(f_vecinos)

    mejor_vecino = vecinos[idx_mejor]

    mejor_valor = f_vecinos[idx_mejor]

    # mover si mejora
    if mejor_valor > fx:

        x = mejor_vecino
        fx = mejor_valor

    else:
        # reducir paso
        step_size *= STEP_DECAY

    # actualizar mejor global
    if fx > best_val:

        best_val = fx
        best_pos = x.copy()

    # guardar historial
    history.append(x.copy())

    # dibujar
    dibujar(t)

# -------------------------------
# Resultado final
# -------------------------------
print(
    f"\nResultado final Hill Climbing -> "
    f"best = ({best_pos[0]:.6f}, "
    f"{best_pos[1]:.6f}), "
    f"f = {best_val:.6f}"
)

plt.ioff()

plt.show()