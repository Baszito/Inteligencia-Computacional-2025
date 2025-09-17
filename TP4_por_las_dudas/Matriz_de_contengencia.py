import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Matriz_de_contengencia(elementos_1, elementos_2):

    filas = len(elementos_1)
    columnas = len(elementos_2)
    
    matriz = np.zeros((filas, columnas))
    
    for i in range(filas):
        for j in range(columnas):
            # Pasamos cada lista a un set para ver coincidencias
            coincidencias = set(elementos_1[i]) & set(elementos_2[j])
            matriz[i, j] = len(coincidencias)
    
    
    print(matriz)
    

Matriz_de_contengencia([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]], [[1, 9, 3, 4, 2, 6], [3, 2, 7, 9, 1, 4], [1, 2, 3, 4, 5, 6], [1, 6, 5, 0, 0, 0], [1, 1, 1, 1, 1, 1], [6, 4, 5, 2, 1, 8]])
