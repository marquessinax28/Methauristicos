import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# Algoritmo de Optimización por Colonia de Hormigas (ACO) para maximizar la función "peaks"

# -------------------------------
# Parámetros configurables  
# -------------------------------

NUM_ITERACIONES = 70   # número de iteraciones principales
NUM_ANTS = 30          # número de hormigas
ARCHIVE_SIZE = 25      # tamaño de memoria de feromonas (número de soluciones almacenadas)
Q = 0.5                # Parametro de selección de feromonas (influencia de la calidad de la solución)
XI = 0.85              # Factor para sigma (dispersión de la distribución normal)
PAUSA = 0.25           # segundos entre actualizaciones (visualización)
RANGO = (-3.0, 3.0)    # dominio de búsqueda (x,y)
GRID_RES = 160         # resolución de malla para la superficie
TRAIL_LENGTH = 6       # longitud del rastro visual de cada solución (0 = sin rastro)
SEED = 42              # semilla para reproducibilidad

# -------------------------------
# Función objetivo: "peaks"
# -------------------------------

def peaks(x, y):
    return 3*(1-x)**2*np.exp(-(x**2) - (y+1)**2) \
           - 10*(x/5 - x**3 - y**5)*np.exp(-x**2 - y**2) \
           - 1/3*np.exp(-(x+1)**2 - y**2)   
           
# -------------------------------
# Inialización 
# -------------------------------

rng = np.random.default_rng(SEED)
lo, li = RANGO
dim = 2

#crear archivo de feromonas (memoria de soluciones)
archive_X = rng.uniform(lo, li, size=(ARCHIVE_SIZE, dim))  # soluciones (x,y)
archive_vals = np.array([peaks(x, y) for x, y in archive_X])  # valores de la función objetivo

# ordenar el archivo por calidad (valor de la función)
order = np.argsort(-archive_vals)  # orden descendente
archive_X = archive_X[order]
archive_vals = archive_vals[order]

# historial para visualización (lista de listas)
history = [ [archive_X[i].copy()] for i in range(ARCHIVE_SIZE) ]

# mejor global 
best_idx = np.argmax(archive_vals)
best_pos = archive_X[best_idx].copy()
best_val = archive_vals[best_idx]

#-------------------------------
# Peparar la malla para visualización
#-------------------------------

Xg = np.linspace(lo, li, GRID_RES)
Yg = np.linspace(lo, li, GRID_RES)
Xmg, Ymg = np.meshgrid(Xg, Yg)
Zmg = peaks(Xmg, Ymg)

plt.ion()
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

cont0 = ax1.contourf(Xmg, Ymg, Zmg, levels=40, cmap='viridis')
cbar = plt.colorbar(cont0, ax=ax1, fraction=0.046, pad=0.04)

cbar.set_label('Valor de funcion')

def dibujar(iteracion, sols, vals, best_pos, best_val, history):
    ax1.clear()
    ax2.clear()
    
    #contorno 2D
    ax1.contourf(Xmg, Ymg, Zmg, levels=40, cmap='viridis')
    ax1.set_xlim(lo, li)
    ax1.set_ylim(lo, li)
    ax1.set_title(f'ACOR - Contorno 2D (Iteración {iteracion+1})')
    
    #Dibuja rastros con degradado de alfa si aplica 
    if TRAIL_LENGTH > 0:
        for i in range(len(history)):
            trail = np.array(history[i][-TRAIL_LENGTH:])  # obtener el rastro reciente
            L = trail.shape[0]
            if L > 1:
                for j in range(L-1):
                    a = 0.15 + 0.85 * (j + 1) / L  # alfa degradado
                    ax1.plot(trail[j:j+2, 0], trail[j:j+2, 1], linewidth=1, alpha=a, color='cyan')
                    
    #Soluciones actuales
    ax1.scatter(sols[:, 0], sols[:, 1], c='blue', s=36, label='Muestras')
    #Mejor global
    ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='Mejor global')
    ax1.legend(loc='upper right', fontsize='small')
    
    #3D
    ax2.plot_surface(Xmg, Ymg, Zmg, cmap='viridis', alpha=0.85, linewidth=0, antialiased=False)
    ax2.set_xlim(lo, li)
    ax2.set_ylim(lo, li)
    ax2.scatter(sols[:, 0], sols[:, 1], vals, c='blue', s=36)
    ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
    ax2.view_init(elev=35, azim=-60)
    
    plt.tight_layout()  
    plt.draw()
    plt.pause(PAUSA)

#-------------------------------
# Funciones auxiliares para ACO
#-------------------------------

def compute_weights(m,q):
    #indice k desde 1..m
    k = np.arange(1, m+1)
    # peso no normalizado
    denom = q * m * np.sqrt(2 * np.pi)
    numer = np.exp(-(k - 1)**2 / (2 * (q * m)**2))
    w = numer / denom
    # Normalizar los pesos para que sumen 1, convirtiéndolos en probabilidades válidas
    w = w / np.sum(w)
    return w

def compute_sigma(archive):
    m, d = archive.shape
    sigmas = np.zeros_like(archive)
    for i in range(m):
        diffs = np.abs(archive[i] - archive)  
        denom = max(1, m-1)  # evitar división por cero
        sigmas[i] = XI * diffs.sum(axis=0) / denom # sumamos a través de las filas (axis=0)
        # si es muy pequeño, usar un valor mínimo para evitar colapsos
        sigmas[i] = np.maximum(sigmas[i], 1e-6)
    return sigmas

#-------------------------------
# Bucle principal ACO
#-------------------------------

for it in range(1, NUM_ITERACIONES + 1):
    m = ARCHIVE_SIZE
    # 1) Calcular pesos y sigmas
    weights = compute_weights(m, Q)
    
    # 2) Generar nuevas soluciones por cada hormiga 
    sigmas = compute_sigma(archive_X)
     
    # 3) Generar nuevo conjunto de soluciones por cada hormiga
    new_sols = np.zeros((NUM_ANTS, dim))
    new_vals = np.zeros(NUM_ANTS)
    
    #para cada ant: seleccionar una solución del archivo con probabilidad proporcional a su peso
    cum_weights = np.cumsum(weights)
    for a in range(NUM_ANTS):
        r = rng.random()
        k = np.searchsorted(cum_weights, r, side='right')
        # generar muestra para cada dimensión
        sample = rng.normal(loc=archive_X[k], scale=sigmas[k]) 
        # clip al domino 
        sample = np.clip(sample, lo, li)
        new_sols[a] = sample
        new_vals[a] = peaks(sample[0], sample[1])
    
    # 4) Actualizar el archivo de feromonas con las nuevas soluciones
    combined_X = np.vstack((archive_X, new_sols))
    combined_vals = np.concatenate((archive_vals, new_vals))
    order = np.argsort(-combined_vals)  # ordenar por valor descendente
    combined_X = combined_X[order][:ARCHIVE_SIZE]  # mantener solo las mejores ARCHIVE_SIZE
    combined_vals = combined_vals[order][:ARCHIVE_SIZE]
    
    #actualizar archivo
    for idx in range(ARCHIVE_SIZE):
        history[idx].append(combined_X[idx].copy())
    
    #actualizar mejor global
    best_idx = np.argmax(combined_vals)
    if combined_vals[best_idx] > best_val:
        best_val = combined_vals[best_idx]
        best_pos = combined_X[best_idx].copy()
        
    # Dibujar el estado actual
    dibujar(it, np.vstack((archive_X, new_sols)), np.concatenate((archive_vals, new_vals)), best_pos, best_val, history)
    
    # CORRECCIÓN CRÍTICA: Actualizar el archivo para la siguiente iteración
    archive_X = combined_X.copy()
    archive_vals = combined_vals.copy()
    
# ---------------------------------    
# Resultados finales
# ---------------------------------

print(f'\nResultado final ACO ---> best = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f = {best_val:.6f}')

plt.ioff()
plt.show()