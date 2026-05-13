
import numpy as np 
import matplotlib.pyplot as plt 
import time
"Teoricamente ya tiene una inicialización aleatoria"
#Funcion peaks
def peaks(x,y):
    return 3*(1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x + 1)**2 - y**2)

#

def hill_climbing(start=None, bounds=(-3,3), step_size=0.3, step_decay=0.5, min_step=1e-3, pause=0.1):
    lo, hi = bounds
    
    if start is None:
        x, y = np.random.uniform(lo,hi), np.random.uniform(lo,hi)
    else:
        x,y = start
    
    fx = peaks(x,y)   
    dirs=[(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0)]
    
    # Crear malla
    x_vals = np.linspace(-3, 3, 200)
    y_vals = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = peaks(X, Y)

    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121)
    cont = ax1.contourf(X, Y, Z, levels=40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title('Hill Climbing - Contorno 2D')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax2.set_title('Hill Climbing - Superficie 3D')

    path = [(x,y,fx)]

    while step_size >= min_step:
        best_neighbor = None
        best_val = fx 
        
        for dx, dy in dirs:
            xn = np.clip(x + dx*step_size, lo, hi)
            yn = np.clip(y + dy*step_size, lo, hi)
            fn = peaks(xn, yn)
            
            if fn > best_val:
                best_val, best_neighbor = fn, (xn,yn)
        
        if best_neighbor:
            x, y = best_neighbor
            fx = best_val
            path.append((x,y,fx))

            ax1.plot([p[0] for p in path], [p[1] for p in path], "w.-")
            ax2.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], "r.-")

            plt.draw()
            plt.pause(pause)
        else:
            step_size *= step_decay

    print(f"Mejor solucion: ({x:.4f}, {y:.4f}), f(x,y)={fx:.4f}")
    plt.ioff()
    plt.show()


#Ejecutar
hill_climbing(pause=0.4)