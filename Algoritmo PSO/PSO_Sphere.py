import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

NUM_ITERACIONES = 60
SWARM_SIZE = 25
W_MAX = 0.9
W_MIN = 0.4
C1 = 1.5
C2 = 1.5
RANGO = (-3.0,3.0)
PAUSA = 0.25
SEED = 42
V_MAX_RATIO = 0.2
TRAIL_LENGHT = 6
GRID_RES = 160

def sphere(x, y):
    return -x**2 + -y**2

#Incialización de la enjambre
rng = np.random.default_rng(SEED)
lo, hi = RANGO
vmax = V_MAX_RATIO * (hi - lo)

#posiciones; SWARM_SIZE * 2 
pos= rng.uniform(lo, hi, size = (SWARM_SIZE, 2))
#velocidades; inicial pequeño 
vel = rng.uniform(-vmax, vmax, size = (SWARM_SIZE, 2))

#mejor personal (pbest) y sus valores
pbest_pos = pos.copy()
pbest_val = np.array([sphere(x,y) for x,y in pbest_pos])

#mejor global (gbest) y su valor
gbest_idx = np.argmax(pbest_val)
gbest_pos = pbest_pos[gbest_idx].copy() 
gbest_val = pbest_val[gbest_idx]

#para dibujar rasrto (opcional)
history = [[pos[i].copy()] for i in range(SWARM_SIZE)]

#Preparar malla y figuras 

X = np.linspace(lo, hi, GRID_RES)
Y = np.linspace(lo, hi, GRID_RES)
Xg, Yg = np.meshgrid(X, Y)
Zg = sphere(Xg, Yg)

plt.ion()
fig = plt.figure(figsize=(13,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

#Crear contorno inicila y una sola barra de color 
cont = ax1.contourf(Xg, Yg, Zg, 40, cmap='viridis')
cbar = plt.colorbar(cont, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Valor de la funcion")

#Funcion para dibujar estado actual 
def dibujar(iteracion):
    ax1.clear()
    ax2.clear() 

    #Redibujar comtorno
    cont = ax1.contourf(Xg, Yg, Zg, 40, cmap='viridis')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title(f"PSO - contorno 2D (Iter {iteracion}/{NUM_ITERACIONES})")

    #Dibujar rastros (solo si aplica)
    if TRAIL_LENGHT > 0:
        for i in range(SWARM_SIZE):
            trail = np.array(history[i][-TRAIL_LENGHT:])
            if trail.shape[0] > 1:
                ax1.plot(trail[:,0], trail[:,1], linewidth=1, alpha=0.6)

    ax1.scatter(pos[:,0], pos[:,1], c='blue', s=30, label='Particulas')
    ax1.scatter(pbest_pos[:,0], pbest_pos[:,1], c='green', s=20, marker='o', label='Pbest')
    ax1.scatter(gbest_pos[0], gbest_pos[1], c='red', s=120, marker='*', label='Gbest')  
    ax1.legend(loc='upper right', fontsize = 'small')

    #Superficie 3D
    ax2.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.8, linewidth=0, antialiased=False)
    ax2.set_title(f"PSO - superficie 3D")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    fitness_curr = np.array([sphere(x,y) for x,y in pos])
    ax2.scatter(pos[:,0], pos[:,1], fitness_curr, c='blue', s=30,) 
    pbest_vals = np.array([sphere(x,y) for x,y in pbest_pos])
    ax2.scatter(pbest_pos[:,0], pbest_pos[:,1], pbest_vals, c='green', s=20)   
    ax2.scatter([gbest_pos[0]], [gbest_pos[1]], [gbest_val], c='red', s=120, marker='*')
    ax2.view_init(elev=35, azim=60)

    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)    
    time.sleep(PAUSA)   

dibujar(0)

for t in range(1,NUM_ITERACIONES + 1):
    #Actualizar inercia (lineal decreciente)
    w = W_MAX - (W_MAX - W_MIN) * (t -1) / max(1, NUM_ITERACIONES - 1)
    #para cada particula actualizar velocidad y posición
    for i in range(SWARM_SIZE):
        r1, r2 = rng.random(2)
        cognitive = C1 * r1 * (pbest_pos[i] - pos[i])
        social = C2 * r2 * (gbest_pos - pos[i]) 
        vel[i] = w * vel[i] + cognitive + social
        #Limitar velocidad
        vel[i] = np.clip(vel[i], -vmax, vmax)

        #actualizar posición
        pos[i] = pos[i] + vel[i]
        #mantener en los limites del dominio
        pos[i,0] = np.clip(pos[i,0], lo, hi)
        pos[i,1] = np.clip(pos[i,1], lo, hi)

        #guardar historial para rastros
        history[i].append(pos[i].copy())

    fitness = np.array([sphere(x,y) for x,y in pos])
    improved_mask = fitness > pbest_val
    if np.any(improved_mask):
        pbest_pos[improved_mask] = pos[improved_mask]
        pbest_val[improved_mask] = fitness[improved_mask]   

    idx = np.argmax(pbest_val)
    if pbest_val[idx] > gbest_val:
        gbest_pos = pbest_pos[idx]
        gbest_val = pbest_val[idx].copy()
    dibujar(t)
print(f"\nResultado final PSO -> gbest: {gbest_pos[0]:.6f},{gbest_pos[1]:.6f} (f =  {gbest_val:.4f})")    
plt.ioff()
plt.show()