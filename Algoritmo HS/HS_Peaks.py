import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------
# Parámetros configurables
# --------------------------------
HMS = 20                # Harmony Memory Size (nº de armonías en HM)
MAX_ITER = 80           # Nº de iteraciones (improvisaciones)
HMCR = 0.95             # Harmony Memory Considering Rate (prob. de tomar valor desde HM)
PAR = 0.3               # Pitch Adjust Rate (prob. de ajustar la nota)
BW = 0.1                # Bandwidth para pitch adjustment (magnitud del ajuste)
PAUSA = 0.25            # segundos entre actualizaciones (visualización)
RANGO = (-3.0, 3.0)     # dominio de búsqueda (x,y)
GRID_RES = 160          # resolución de la malla para la superficie
TRAIL_LENGTH = 8        # longitud del rastro visual de cada armonía (0 = sin rastro)
SEED = 2024             # semilla para reproducibilidad

# --------------------------------
# Función objetivo: "peaks"
# --------------------------------
def peaks(x, y):
    return 3*(1-x)**2 * np.exp(-(x**2)-(y+1)**2) \
         - 10*(x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
         - 1/3 * np.exp(-(x+1)**2-y**2)

# --------------------------------
# Inicialización
# --------------------------------
rng = np.random.default_rng(SEED)
lo, hi = RANGO
dim = 2 # x, y

# Inicializar Harmony Memory (HM) con HMS armonías aleatorias
HM = rng.uniform(lo, hi, size=(HMS, dim))
HM_vals = np.array([peaks(x, y) for x, y in HM])

# historial para rastro (lista de listas)
history = [[HM[i].copy()] for i in range(HMS)]

# mejor global inicial
best_idx = np.argmax(HM_vals)
best_pos = HM[best_idx].copy()
best_val = HM_vals[best_idx]

# --------------------------------
# Preparar malla y figuras (crear colorbar UNA vez)
# --------------------------------
Xg = np.linspace(lo, hi, GRID_RES)
Yg = np.linspace(lo, hi, GRID_RES)
Xmg, Ymg = np.meshgrid(Xg, Yg)
Zmg = peaks(Xmg, Ymg)

plt.ion()
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121) # 2D contorno
ax2 = fig.add_subplot(122, projection='3d') # 3D superficie

# contorno inicial y barra de color UNA sola vez
cont0 = ax1.contourf(Xmg, Ymg, Zmg, 40, cmap="viridis")
cbar = plt.colorbar(cont0, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Valor de la función")

def dibujar(iteracion, HM, HM_vals, best_pos, best_val, history):
    # Limpiar e insertar gráficos (la colorbar fue creada fuera)
    ax1.clear()
    ax2.clear()

    # redibujar contorno (cbar ya existe y evita acumular barras)
    ax1.contourf(Xmg, Ymg, Zmg, 40, cmap="viridis")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title(f"Harmony Search Contorno 2D (Iter {iteracion}/{MAX_ITER})")

    # dibujar rastros con degradado alfa (si aplica)
    if TRAIL_LENGTH > 0:
        for i in range(len(history)):
            trail = np.array(history[i][-TRAIL_LENGTH:])
            L = trail.shape[0]
            if L > 1:
                # segmentos con alpha creciente (más reciente más opaco)
                for k in range(L-1):
                    a = 0.15 + 0.85 * (k+1)/L
                    ax1.plot(trail[k:k+2, 0], trail[k:k+2, 1], linewidth=1.2, alpha=a, color='cyan')

    # dibujar HM actual: puntos azules
    ax1.scatter(HM[:,0], HM[:,1], c='blue', s=36, label='HM (armonías)')
    # marcar la mejor
    ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='best global')
    ax1.legend(loc='upper right', fontsize='small')

    # Superficie 3D
    ax2.plot_surface(Xmg, Ymg, Zmg, cmap="viridis", alpha=0.86, linewidth=0, antialiased=False)
    ax2.set_title("Harmony Search Superficie 3D")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.scatter(HM[:,0], HM[:,1], HM_vals, c='blue', s=36)
    ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
    ax2.view_init(elev=35, azim=-60)

    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)
    time.sleep(PAUSA)

# --------------------------------
# Bucle principal de HS (improvisaciones)
# --------------------------------
for it in range(1, MAX_ITER + 1):
    # generar una nueva armonía (vector) siguiendo HMCR y PAR
    new = np.empty(dim)
    for d in range(dim):
        if rng.random() < HMCR:
            # tomar valor aleatorio desde HM (memory consideration)
            idx = rng.integers(0, HMS)
            val = HM[idx, d]
            # pitch adjustment con probabilidad PAR
            if rng.random() < PAR:
                # ajuste pequeño en el vecindario
                delta = BW * (hi - lo) # escala del ajuste
                # ajuste uniforme en [-delta, delta]
                val = val + rng.uniform(-delta, delta)
        else:
            # selección aleatoria en el dominio
            val = rng.uniform(lo, hi)
        
        # asegurar límites
        new[d] = np.clip(val, lo, hi)

    # evaluar nueva armonía
    new_val = peaks(new[0], new[1])

    # si la nueva es mejor que la peor en HM, reemplazarla
    worst_idx = np.argmin(HM_vals)
    if new_val > HM_vals[worst_idx]:
        HM[worst_idx] = new
        HM_vals[worst_idx] = new_val
        # actualizar historial: resetear historial del reemplazado con la nueva
        history[worst_idx].append(new.copy())
    else:
        # si no reemplaza, igualmente agregar al historial de una armonia aleatoria
        # (esto hace que los rastros muestren actividad)
        r_idx = rng.integers(0, HMS)
        history[r_idx].append(HM[r_idx].copy())
    
    # actualizar mejor global
    cur_best_idx = np.argmax(HM_vals)
    if HM_vals[cur_best_idx] > best_val:
        best_val = HM_vals[cur_best_idx]
        best_pos = HM[cur_best_idx].copy()
    
    # dibujar estado actual
    dibujar(it, HM, HM_vals, best_pos, best_val, history)

# --------------------------------
# Resultado final
# --------------------------------
print(f"\nResultado final HS -> best = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f = {best_val:.6f}")

plt.ioff()
plt.show()