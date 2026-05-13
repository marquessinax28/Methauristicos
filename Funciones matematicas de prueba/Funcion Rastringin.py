import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

A = 10
Z = A * 2 + (X**2 - A * np.cos(2 *np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')

ax.set_title("Función Rastrigin")
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")
ax.set_zlabel("f(x, y)")
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(elev=30, azim=45)
plt.show()

plt.figure(figsize=(7, 6))
contour = plt.contour(X, Y, Z, levels=50, cmap='jet')
plt.colorbar(contour)
plt.title("Curvas de nivel de la función Rastrigin")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()