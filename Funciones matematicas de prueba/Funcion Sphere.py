import numpy as np
import matplotlib.pyplot as pltp


x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

Z = X**2 + Y**2

fig = pltp.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')

ax.set_title("Función Sphere: f(x, y) = x² + y²")
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")
ax.set_zlabel("f(x, y)")
fig.colorbar(surf, shrink=0.5, aspect=10)

ax.view_init(elev=30, azim=45)
pltp.show()
