import numpy as np 
import matplotlib.pyplot as plt 
import time

#Funcion peaks
def peaks(x,y):
    return 3*(1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x + 1)**2 - y**2)
           
           
# Busqueda aleatoria 

def busqueda_aleatoria(num_iter =50, rango =(-3,3), pausa=0.2):
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    z = peaks(X, Y)
    
    plt.ion()  # Modo interactivo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    #contorno
    cont = ax1.contourf(X, Y, z, levels=40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title('Busqueda Aleatoria - Contorno 2D')
    
    #Superficie 3D
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, z, cmap='viridis', alpha=0.8)
    ax2.set_title('Busqueda Aleatoria - Superficie 3D')
    
    mejor_val = -np.inf
    mejor_punto = None
    
    for i in range(num_iter):    
        x = np.random.uniform(rango[0], rango[1])
        y = np.random.uniform(rango[0], rango[1])
        val = peaks(x, y)
        
        #Guardar mejor punto
        if val > mejor_val:
            mejor_val, mejor_punto =val,(x, y)    

        
        #Dibujar paso en 2D y 3D
        ax1.scatter(x, y, color='blue', s=30)
        ax2.scatter(x, y, val, color='blue', s=30)

        #Resaltar mejor actual 
        ax1.scatter(mejor_punto[0], mejor_punto[1], color='red', s=60)
        ax2.scatter(mejor_punto[0], mejor_punto[1], mejor_val, color='red', s=80)
        
        plt.draw()
        plt.pause(pausa) #pausa entre pasos
        time.sleep(pausa)
    
    print(f"Mejor solucion: {mejor_punto}, f(x,y)={mejor_val:.4f}")
    plt.ioff()
    plt.show()
    
#Ejecutar busqueda aleatoria
busqueda_aleatoria(num_iter=50, pausa=0.3)
