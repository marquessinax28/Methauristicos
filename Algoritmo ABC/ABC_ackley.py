import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Parámetros configurables
# -------------------------------
NUM_ITERACIONES = 80    # número de iteraciones principales
NUM_FUENTES = 20        # número de fuentes (food sources) = número de employed bees
LIMIT = 15              # número de intentos sin mejorar antes de convertir en scout
PAUSA = 0.25            # segundos entre actualizaciones (visualización)
RANGO = (-3.0, 3.0)     # dominio de búsqueda (x,y)
GRID_RES = 160          # resolución de malla para la superficie
TRAIL_LENGTH = 6        # longitud del rastro visual de cada fuente (0 = sin rastro)
SEED = 123              # semilla para reproducibilidad
SEED_RANDOM = np.random.default_rng(SEED)

# -------------------------------
# Función objetivo: "ackley"
# -------------------------------
def ackley(x, y):
    A = 20
    B = 0.2
    C = 2 * np.pi
    
    term1 = -A * np.exp(-B * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(C * x) + np.cos(C * y)))
    
    return term1 + term2 + A + np.exp(1)

# -------------------------------
# Inicialización de fuentes
# -------------------------------
lo, hi = RANGO
rng = SEED_RANDOM

# posiciones de las fuentes (NUM_FUENTES x 2)
sources = rng.uniform(lo, hi, size=(NUM_FUENTES, 2))

# valor de la función y fitness asociado
values = np.array([ackley(x, y) for x, y in sources])

# Para ABC a veces se usa fitness = 1/(1+f) si es minimización; aquí maximizamos ackley,
# así que transformamos para probabilidad de selección (positiva)
# Ajuste para probabilidades:
def fitness_to_prob(vals):
    # shift to positive and avoid zeros
    shifted = vals - vals.min() + 1e-9
    probs = shifted / shifted.sum()
    return probs

# contador de intentos sin mejora (trial) para cada fuente
trials = np.zeros(NUM_FUENTES, dtype=int)

# historial para rastro (lista de listas)
history = [ [sources[i].copy()] for i in range(NUM_FUENTES) ]

# mantener la mejor solución global
best_idx = np.argmax(values)
best_pos = sources[best_idx].copy()
best_val = values[best_idx]

# -------------------------------
# Preparar malla y figuras (una sola colorbar)
# -------------------------------
X = np.linspace(lo, hi, GRID_RES)
Y = np.linspace(lo, hi, GRID_RES)
Xg, Yg = np.meshgrid(X, Y)
Zg = ackley(Xg, Yg)

plt.ion()
fig = plt.figure(figsize=(13,6))
ax1 = fig.add_subplot(121)                    # 2D contorno
ax2 = fig.add_subplot(122, projection='3d')   # 3D superficie

# crear contorno inicial y barra de color UNA sola vez
cont = ax1.contourf(Xg, Yg, Zg, 40, cmap="viridis")
cbar = plt.colorbar(cont, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Valor de la función")

def dibujar(iteracion):
    ax1.clear()
    ax2.clear()

    # redibujar contorno (la barra de color sigue siendo la misma figura)
    cont = ax1.contourf(Xg, Yg, Zg, 40, cmap="viridis")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title(f"ABC - Contorno 2D (Iter {iteracion}/{NUM_ITERACIONES})")

    # dibujar rastros si aplica
    if TRAIL_LENGTH > 0:
        for i in range(NUM_FUENTES):
            trail = np.array(history[i][-TRAIL_LENGTH:])
            if trail.shape[0] > 1:
                ax1.plot(trail[:,0], trail[:,1], linewidth=1, alpha=0.6)

    # fuentes actuales (azul), mejores locales (verde pequeño) y global (rojo estrella)
    ax1.scatter(sources[:,0], sources[:,1], c='blue', s=35, label='fuentes')
    ax1.scatter(sources[:,0], sources[:,1], c='blue', s=35)  # puntos de fuentes
    ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='best global')

    ax1.legend(loc='upper right', fontsize='small')

    # superfície 3D
    ax2.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.85, linewidth=0, antialiased=False)
    ax2.set_title("ABC - Superficie 3D")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    current_vals = np.array([ackley(x,y) for x,y in sources])
    ax2.scatter(sources[:,0], sources[:,1], current_vals, c='blue', s=35)
    ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
    ax2.view_init(elev=35, azim=-60)

    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)
    time.sleep(PAUSA)

# -------------------------------
# Bucle principal ABC
# -------------------------------
for t in range(1, NUM_ITERACIONES + 1):
    # ---------- 1) Employed bees ----------
    # cada employed bee intenta mejorar su fuente actual
    for i in range(NUM_FUENTES):
        # elegir una fuente distinta k aleatoria
        k = rng.integers(0, NUM_FUENTES-1)
        if k >= i:
            k += 1
        # generar phi en [-1,1] para cada dimensión
        phi = rng.uniform(-1.0, 1.0, size=2)
        # construir nueva solución v_ij = x_ij + phi*(x_ij - x_kj)
        v = sources[i] + phi * (sources[i] - sources[k])
        # acotar
        v = np.clip(v, lo, hi)
        fv = ackley(v[0], v[1])
        # si mejora, reemplazar y resetear trials
        if fv > values[i]:
            sources[i] = v
            values[i] = fv
            trials[i] = 0
        else:
            trials[i] += 1

        # guardar historial
        history[i].append(sources[i].copy())

    # actualizar mejor global
    idx = np.argmax(values)
    if values[idx] > best_val:
        best_val = values[idx]
        best_pos = sources[idx].copy()

    # ---------- 2) Onlooker bees ----------
    # calcular probabilidades de selección basadas en fitness positivo
    probs = fitness_to_prob(values)
    # cada onlooker realiza una prueba; se suelen hacer NUM_FUENTES onlookers (pueden variar)
    onlookers = NUM_FUENTES
    count = 0
    i = 0
    while count < onlookers:
        # seleccionar fuente i por ruleta (probabilístico)
        sel = rng.choice(np.arange(NUM_FUENTES), p=probs)
        # intentar mejorar la fuente sel de la misma forma que employed
        k = rng.integers(0, NUM_FUENTES-1)
        if k >= sel:
            k += 1
        phi = rng.uniform(-1.0, 1.0, size=2)
        v = sources[sel] + phi * (sources[sel] - sources[k])
        v = np.clip(v, lo, hi)
        fv = ackley(v[0], v[1])
        if fv > values[sel]:
            sources[sel] = v
            values[sel] = fv
            trials[sel] = 0
        else:
            trials[sel] += 1

        history[sel].append(sources[sel].copy())

        count += 1
        i += 1

    # actualizar mejor global
    idx = np.argmax(values)
    if values[idx] > best_val:
        best_val = values[idx]
        best_pos = sources[idx].copy()

    # ---------- 3) Scout bees ----------
    # si una fuente no mejoró en 'LIMIT' intentos, reemplazarla aleatoriamente
    for i in range(NUM_FUENTES):
        if trials[i] >= LIMIT:
            # reemplazar por nueva fuente aleatoria
            sources[i] = rng.uniform(lo, hi, size=2)
            values[i] = ackley(sources[i][0], sources[i][1])
            trials[i] = 0
            history[i].append(sources[i].copy())

    # actualizar mejor global otra vez por si acaso
    idx = np.argmax(values)
    if values[idx] > best_val:
        best_val = values[idx]
        best_pos = sources[idx].copy()

    # dibujar estado actual
    dibujar(t)

# -------------------------------
# Resultado final
# -------------------------------
print(f"\nResultado final ABC -> best = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f = {best_val:.6f}")

plt.ioff()
plt.show()