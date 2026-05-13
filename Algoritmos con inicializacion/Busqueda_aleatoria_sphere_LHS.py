import numpy as np
import matplotlib.pyplot as plt
import time


def sphere(x, y):
    return x**2 + y**2


def busqueda_aleatoria(num_iter=50, rango=(-5,5), pausa=0.2):

    x_vals = np.linspace(rango[0], rango[1], 200)
    y_vals = np.linspace(rango[0], rango[1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = sphere(X, Y)

    plt.ion()
    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(121)
    cont = ax1.contourf(X, Y, Z, levels=40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title("Sphere - Contorno 2D")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax2.set_title("Sphere - Superficie 3D")

    mejor_val = np.inf
    mejor_punto = None

    # LHS - Generacion de poblacion inicial
    lo, hi = rango
    indices = np.arange(-(num_iter - 1) / 2, (num_iter - 1) / 2 + 1)
    matriz = np.zeros((num_iter, 2))
    for dim in range(2):
        matriz[:, dim] = np.random.permutation(indices)
    ajuste = np.random.rand(num_iter, 2)
    norm = (matriz + (num_iter - 1) / 2 + ajuste) / num_iter
    poblacion = norm * (hi - lo) + lo

    for i in range(num_iter):

        x, y = poblacion[i, 0], poblacion[i, 1]
        val = sphere(x, y)

        if val < mejor_val:
            mejor_val = val
            mejor_punto = (x, y)

        ax1.scatter(x, y, color='blue', s=30)
        ax2.scatter(x, y, val, color='blue', s=30)

        ax1.scatter(mejor_punto[0], mejor_punto[1], color='red', s=60)
        ax2.scatter(mejor_punto[0], mejor_punto[1], mejor_val, color='red', s=80)

        plt.pause(pausa)

    print(f"Mejor solución encontrada: {mejor_punto}")
    print(f"Valor mínimo: {mejor_val:.6f}")

    plt.ioff()
    plt.show()


busqueda_aleatoria(num_iter=60, pausa=0.3)