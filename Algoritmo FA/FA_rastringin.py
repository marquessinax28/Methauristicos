import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

#Parametros configurables
NUM_ITERACIONES = 80
NUM_FIREFLIES = 30 
BETA0 = 1.0
GAMMA = 1.0
ALPHA = 0.2
ALPHA_DECAY = 0.98
PAUSA = 0.18 
RANGO = (-3.0, 3.0)
GRID_RES = 160
TRAIL_LENGTH = 8
SEED = 2025
TRAND_SCALE = 1.0

#Funcion objetivo rastrigin
def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))  

#Inicialización de las luciernagas
rng = np.random.default_rng(SEED)
lo,hi = RANGO
dim = 2
scale_noise = TRAND_SCALE * (hi - lo)

#Posiciones de las luciernagas
X = rng.uniform(lo, hi, size=(NUM_FIREFLIES, dim))
#Valores y "brillo"
vals = np.array([rastrigin(x,y) for x,y in X])

#Historial para trazos
history = [[X[i].copy()] for i in range(NUM_FIREFLIES)]

#Mejor global
best_idx = np.argmax(vals)
best_pos = X[best_idx].copy()
best_val = vals[best_idx]

#Preparar mallla y figuras 
Xg = np.linspace(lo, hi, GRID_RES)
Yg = np.linspace(lo, hi, GRID_RES)
Xmg, Ymg = np.meshgrid(Xg, Yg)
Zmg = rastrigin(Xmg, Ymg)

plt.ion()
fig = plt.figure(figsize=(13,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

#Contorno incial colobar una vez para evitar acomular barras
cont0 = ax1.contourf(Xmg, Ymg, Zmg, 40, cmap='viridis')
cbar = plt.colorbar(cont0, ax=ax1, fraction = 0.046, pad=0.04)
cbar.set_label('Valor de la función')
 
def dibujar (iteracion, X, vals, best_pos, best_val, history, alpha_vis):
    ax1.clear()
    ax2.clear()
    
    #Contorno
    cont = ax1.contourf(Xmg, Ymg, Zmg, 40, cmap='viridis')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)    
    ax1.set_title(f'FireFly Algorithm - Contorno 2D  {iteracion}/{NUM_ITERACIONES}')

    #Dibujar rastros con degradado de alpha solo si aplica
    if TRAIL_LENGTH > 0:
        for i in range(NUM_FIREFLIES):
            trail = np.array(history[i][-TRAIL_LENGTH:])
            L = trail.shape[0]
            if L > 1:
                for k in range(L-1):
                    a = alpha_vis * (k+1) / L
                    ax1.plot(trail[k:k+2,0], trail[k:k+2,1], color='cyan', alpha=a, linewidth=1.2)

    #Posiciones actuales
    ax1.scatter(X[:,0], X[:,1], c='blue', s=36, label='Luciernagas')
    ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='Mejor Global')
    ax1.legend(loc='upper right', fontsize='small')

    #Superficies 3D
    ax2.plot_surface(Xmg, Ymg, Zmg, cmap='viridis', alpha=0.86, linewidth=0, antialiased=False)
    ax2.set_title(f'FireFly Algorithm - Superficie 3D')
    ax2.scatter(X[:,0], X[:,1], vals, c='blue', s=36)
    ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
    ax2.view_init(elev=35, azim=60)

    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)
    time.sleep(PAUSA)

#Bucle principal del algoritmo de las luciernagas
alpha_curr = ALPHA
for t in range(1, NUM_ITERACIONES + 1):
    order = np.argsort(vals)

    for idx_i in range(NUM_FIREFLIES):
        i = order[idx_i]
        xi = X[i].copy()

        for idx_j in range(0, idx_i):
            j = order[idx_j]
            xj = X[j]
            r = np.linalg.norm(xi - xj)
            beta = BETA0 * np.exp(-GAMMA * r**2)
            noise = rng.normal(0, 1, size=dim)
            xi = xi + beta * (xj - xi) + alpha_curr * noise * (scale_noise / 0.2)
            xi = np.clip(xi, lo, hi)

        X[i] = xi
        vals[i] = rastrigin(xi[0], xi[1])
        history[i].append(X[i].copy())

    alpha_curr *= ALPHA_DECAY

    cur_best = np.argmax(vals)
    if vals[cur_best] > best_val:
        best_val = vals[cur_best]
        best_pos = X[cur_best].copy()

    dibujar(t, X, vals, best_pos, best_val, history, alpha_vis=0.9)

print(f"\nResultado final FA -> best = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f = {best_val:.6f}")
plt.ioff()
plt.show()