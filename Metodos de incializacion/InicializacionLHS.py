import numpy as np
import matplotlib.pyplot as plt

numPuntos = 60
numDim = 2

randoIndices = np.arange(-(numPuntos -1) / 2, (numPuntos -1) / 2 + 1 )

matrizPerm = np.zeros((numPuntos, numDim))

for dim in range(numDim):
    matrizPerm[:, dim] = np.random.permutation(randoIndices)
    

ajusteRand = np.random.rand(numPuntos, numDim)

puntosNorm = (matrizPerm + (numPuntos- 1 )/ 2 + ajusteRand) / numPuntos

plt.plot(puntosNorm[:, 0], puntosNorm[:, 1], 'o')

plt.xticks(np.arange(0, 1 + 1/numPuntos, 1/numPuntos))
plt.yticks(np.arange(0, 1 + 1/numPuntos, 1/numPuntos))


puntosEsc = np.zeros_like(puntosNorm)
puntosEsc[:, 0] = puntosNorm[:, 0] * 6-3
puntosEsc[:, 1] = puntosNorm[:, 1] * 6-3

plt.grid(True)
plt.show()

puntosSinRand = (matrizPerm + (numPuntos - 1) / 2) / numPuntos

matrizTrans = np.array([[3, 1/2],[2, 1/2]])
