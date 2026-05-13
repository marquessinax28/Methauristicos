import numpy as np
import matplotlib.pyplot as plt

# ==========================
# FUNCIÓN SPHERE
# ==========================
def sphere(x, y):
    return x**2 + y**2


def hill_climbing():
    
    bounds = (-5, 5)
    lo, hi = bounds
    
    # LHS - Generacion del punto inicial
    indices = np.arange(-(1 - 1) / 2, (1 - 1) / 2 + 1)
    matriz = np.zeros((1, 2))
    for dim in range(2):
        matriz[:, dim] = np.random.permutation(indices)
    ajuste = np.random.rand(1, 2)
    norm = (matriz + (1 - 1) / 2 + ajuste) / 1
    poblacion = norm * (hi - lo) + lo
    x, y = poblacion[0, 0], poblacion[0, 1]
    fx = sphere(x,y)

    step_size = 0.5
    step_decay = 0.5
    min_step = 1e-3
    pause = 0.2

    dirs = [(dx,dy) for dx in (-1,0,1)
                     for dy in (-1,0,1)
                     if not (dx==0 and dy==0)]

    x_vals = np.linspace(lo, hi, 200)
    y_vals = np.linspace(lo, hi, 200)
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

    path = [(x,y,fx)]

    while step_size >= min_step:

        best_neighbor = None
        best_val = fx

        for dx, dy in dirs:
            xn = np.clip(x + dx*step_size, lo, hi)
            yn = np.clip(y + dy*step_size, lo, hi)
            fn = sphere(xn, yn)

            if fn < best_val:   
                best_val = fn
                best_neighbor = (xn, yn)

        if best_neighbor:
            x, y = best_neighbor
            fx = best_val
            path.append((x,y,fx))

            ax1.plot([p[0] for p in path],
                     [p[1] for p in path], "w.-")
            ax2.plot([p[0] for p in path],
                     [p[1] for p in path],
                     [p[2] for p in path], "r.-")

            plt.pause(pause)
        else:
            step_size *= step_decay

    print(f"Mejor solución encontrada: ({x:.4f}, {y:.4f})")
    print(f"Valor mínimo: {fx:.6f}")

    plt.ioff()
    plt.show()


hill_climbing()