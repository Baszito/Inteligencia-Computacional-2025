# Resumen de k-medias

# 1) Se elige un numero k de centroides
# 2) se los inicializa de manera aleatoria
# 3) Por cada puntito x se mira cual centroide es el mas cercano
# 4) Se actualiza el centroide yendo hacia el promedio de los x dentro de el.

# Es decir se computa cuantos x tiene mas cerca, se los mete en una bolsa y se hace el promedio de esa bolsa
# y en la siguiente iteracion se clava ahi ese centroide.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class kmeans:
    def __init__(self, path_datos, k, max_epocas=1000):
        self.X = pd.read_csv(path_datos, header=None).values
        self.ks = []
        self.max_epocas = max_epocas
        self.S = {}
        for i in range(0, k):
            self.ks.append(self.generar_vector())
            # Por cada centroide k inicializamos un vector en su diccionario S
            self.S[i] = []
        
    def entradas(self):
        return self.X.shape[1]
    def generar_vector(self) -> np.ndarray:
        """
        Genera un vector de dimensión R^(n_entradas) 
        con valores aleatorios en el rango [-0.5, 0.5].

        Parámetros:
            n_entradas (int): cantidad de entradas del vector

        Retorna:
            np.ndarray: vector de tamaño (n_entradas,)
        """
        return np.random.uniform(-0.5, 0.5, size=self.entradas())
    def graficar(self):
        plt.scatter(self.X[:, 0], self.X[:, 1])
        msize = 10
        for c_k in self.ks:
            plt.plot(c_k[0], c_k[1], '*', markersize=msize, color='red')
            msize += 3
        plt.show()
    def entrenar(self):

        for k in range(0, self.max_epocas):
            
            for i in range(0, self.X.shape[0]):
                d = self.X[i]
                print('Dato d: ', d)
                # Para el dato 'd' computamos el centroide mas cercano
                dist_min = 999999
                c_mas_cercano = 0
                ind_c = 0
                
                for c_k in self.ks:
                    dis = np.linalg.norm(d - c_k) 
                    if dis < dist_min:
                        #c_mas_cercano = c_k
                        c_mas_cercano = ind_c
                        dist_min = dis
                    ind_c += 1
                # A ese centroide le asignamos el indice del dato x
                self.S[c_mas_cercano].append(i)
            # Una vez habiendo iterado todos los datos y sacando los cercanos, iteramos por todos los centroides
            ind_c = 0
            for c_k in self.ks:
                # Por cada centroide, iteramos sus x

                pos = np.zeros(self.ks[0].shape)

                # recordemos que x es un indice
                for x in self.S[ind_c]:

                    pos = pos + self.X[x]
                #if len(self.S[ind_c]) > 0:
                pos = pos / len(self.S[ind_c])
                self.ks[ind_c] = pos
                # Ahora le asignamos al centroide esta posicion
                #self.ks[ind_c] = pos
                ind_c += 1
            # Finalmente limpiamos los vectores en S
            ind_c = 0
            for c_k in self.ks:
                self.S[ind_c] = []
                ind_c += 1

p = kmeans('TP4/probando.csv', 3)
p.entrenar()
p.graficar()