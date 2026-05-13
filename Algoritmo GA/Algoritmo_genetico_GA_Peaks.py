#Algoritmo genetico methauristico
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del algoritmo genético
NUM_GENERACIONES = 30
TAM_POBLACION = 20
PROB_CRUCE = 0.8
PROB_MUTACION = 0.2
RANGO = (-3, 3)
PAUSA = 0.5

def peak(x, y):
    return 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x+1)**2 - y**2)


def algortimo_genetico():
    lo, hi = RANGO
    poblacion = np.random.uniform(lo, hi, size=(TAM_POBLACION, 2))

    X = np.linspace(lo, hi, 200)
    Y = np.linspace(lo, hi, 200)
    X, Y = np.meshgrid(X, Y)
    Z = peak(X, Y)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    cont = ax1.contourf(X, Y, Z, 40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title('Algoritmo Genético - Contorno 2D')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax2.set_title('Algoritmo Genético - Superficie 3D')

    mejor_sol = None
    mejor_val = -np.inf

    for gen in range(NUM_GENERACIONES):
        # Evaluar la aptitud de cada individuo
        fitness = np.array([peak (ind[0], ind[1]) for ind in poblacion])

        idx_best = np.argmax(fitness)
        if fitness[idx_best] > mejor_val:
            mejor_val = fitness[idx_best]
            mejor_sol = poblacion[idx_best]
        
        ax1.clear()
        cont = ax1.contourf(X, Y, Z, 40, cmap='viridis')
        ax1.set_title(f"GA - Cntorno 2D - (Genracion {gen+1})")
        ax1.scatter(poblacion[:,0], poblacion[:,1], color='blue', s=30)
        ax1.scatter(mejor_sol[0], mejor_sol[1], color='red', s=80, marker='*')

        ax2.clear()
        ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax2.set_title(f"GA - Superficie 3D - Generacion({gen+1})")
        ax2.scatter(poblacion[:, 0], poblacion[:, 1], fitness, color='blue', s=30)
        ax2.scatter(mejor_sol[0], mejor_sol[1], mejor_val, color='red', s=80, marker='*')

        plt.draw()
        plt.pause(PAUSA)
        time.sleep(PAUSA)

        # Selección por torneo
        fitness_pos = fitness - fitness.min() + 1e-6
        probs = fitness_pos / fitness_pos.sum()
        padres_idx = np.random.choice(len(poblacion), size=len(poblacion), p=probs)
        padres= poblacion[padres_idx]
        # Cruce
        descendencia = []
        for i in range(0,len(padres),2):
            p1, p2 = padres[i], padres[(i+1) % len(padres)]
            if np.random.rand() < PROB_CRUCE:
                alpha = np.random.rand()
                h1 = alpha * p1 + (1 - alpha) * p2
                h2 = alpha * p2 + (1 - alpha) * p1
                descendencia.extend([h1, h2])
            else:
                descendencia.extend([p1, p2])
        poblacion = np.array(descendencia)

        for ind in poblacion:
            if np.random.rand() < PROB_MUTACION:
                ind += np.random.uniform(-0.3, 0.3, size=2)
                ind[0] = np.clip(ind[0], lo, hi)
                ind[1] = np.clip(ind[1], lo, hi)

    print(f"Mejor solución encontrada: ({mejor_sol}, f(x,y)={mejor_val:.4f})")
    plt.ioff()
    plt.show()

algortimo_genetico()