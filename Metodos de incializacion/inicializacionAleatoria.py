import numpy as np
import matplotlib.pyplot as plt

semilla = 42
x = np.random.rand(semilla)  # 60 puntos aleatorios en el rango [0, 1]
y = np.random.rand(semilla)

semilla = 42  # Establecer una semilla para reproducibilidad
np.random.seed(semilla)
plt.scatter(x, y)
plt.grid(True)

plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Gráfico de Dispersión')

plt.show()