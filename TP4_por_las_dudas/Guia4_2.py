# Resumen de k-medias

# 1) Se elige un numero k de centroides
# 2) se los inicializa de manera pseudo-aleatoria
# 3) Por cada puntito x se mira cual centroide es el mas cercano
# 4) Se actualiza el centroide yendo hacia el promedio de los x dentro de el.

# Es decir se computa cuantos x tiene mas cerca, se los mete en una bolsa y se hace el promedio de esa bolsa
# y en la siguiente iteracion se clava ahi ese centroide.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
#from Guia4_1 import SOM
class SOM:
    def __init__(self, path_datos, nro_filas_neuronas, nro_columnas_neuronas, velocidad_entrenamiento, velocidad_entrenamiento_final = 0.01, max_epocas=1000):
        self.X = pd.read_csv(path_datos, header=None).values
        self.X = self.X[:, :-5]#Le sacamos las salidas + 2 entradas mas, solo tomando los mismos 2 datos que en el K-means
        self.velocidad_entrenamiento = velocidad_entrenamiento
        self.velocidad_entrenamiento_inicial = velocidad_entrenamiento
        self.velocidad_entrenamiento_final = velocidad_entrenamiento_final
        self.max_epocas = max_epocas

        self.mapa_neuronas = []

        self.f = nro_filas_neuronas
        self.c = nro_columnas_neuronas
        for i in range(0, nro_filas_neuronas):
            self.mapa_neuronas.append([])
            for j in range(0, nro_columnas_neuronas):
                self.mapa_neuronas[-1].append(self.generar_vector())
        self.figure, self.axis = plt.subplots(2)

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
    counter = 0
    #def construir_vecindad(self, radio, i_ganador, j_ganador):
    #    for j in range(0, 2*radio):
    #        for i in range()
    def esta_en_la_vecindad(self, radio, i_ganador, j_ganador, i, j):
        bias = 0.1
        radio = radio + bias
        A = np.array([i_ganador, j_ganador + radio])
        B = np.array([i_ganador + radio, j_ganador])
        P = np.array([abs(i-i_ganador) + i_ganador, abs(j-j_ganador) + j_ganador])
        d = (B[0] - A[0])*(P[1] - A[1]) - (B[1] - A[1])*(P[0] - A[0])
        if d >= 0:
            return False
        else:
            return True
    def get_vecindad(self, radio, i_ganador, j_ganador):
        vecindad = []
        js_vecindad = []
        for i in range(0, self.f):
            for j in range(0, self.c):
                if i==i_ganador and j==j_ganador:
                    continue
                if self.esta_en_la_vecindad(radio, i_ganador, j_ganador, i, j):
                    vecindad.append([i, j])
        return vecindad
    def entradas(self):
        return self.X.shape[1]
    def actualizar(self, x, i, j):
        self.mapa_neuronas[i][j] = self.mapa_neuronas[i][j] + self.velocidad_entrenamiento * (x - self.mapa_neuronas[i][j])
    def neurona_ganadora(self, entrada):
        dist_minima = 1000000
        i_ganador = -999
        j_ganador = -999
        for i in range(0, self.f):
            for j in range(0, self.c):
                d = np.linalg.norm(entrada - self.mapa_neuronas[i][j])
                if (d<=dist_minima):
                    i_ganador = i
                    j_ganador = j
                    dist_minima = d
        return [i_ganador, j_ganador]
    def get_pesos(self):
        puntos_x = []
        puntos_y = []
        for i in range(0, self.f):
            for j in range(0, self.c):
                puntos_x.append(self.mapa_neuronas[i][j][0])
                puntos_y.append(self.mapa_neuronas[i][j][1])
        return (puntos_x, puntos_y)

    def entrenar(self):
        radio = np.floor(self.f/2) # Empezamos con un radio igual a la mitad de nuestras filas
        for k in range(0, self.max_epocas):
            self.velocidad_entrenamiento = self.velocidad_entrenamiento_inicial + k*((self.velocidad_entrenamiento_final - self.velocidad_entrenamiento_inicial)/self.max_epocas)
            radio = np.floor(self.f/2 + k*(1 - np.floor(self.f/2))/self.max_epocas)
            print('RADIO: ', radio)
            print('VEL: ', self.velocidad_entrenamiento)
            self.axis[0].cla()
            self.axis[1].cla()
            time.sleep(0.01)

            dato_random_i = np.random.randint(self.X.shape[0])
            dato_random = self.X[dato_random_i, :]
            [i_ganador, j_ganador] = self.neurona_ganadora(dato_random)
            vecindad = self.get_vecindad(radio, i_ganador, j_ganador)
            self.axis[1].plot(i_ganador, j_ganador, 's')
            for i in range(0, self.f):
                for j in range(0, self.c):
                    self.axis[1].plot(i, j, 'o')
            for v in vecindad:
                self.axis[0].plot([self.mapa_neuronas[v[0]][v[1]][0], self.mapa_neuronas[i_ganador][j_ganador][0]], [self.mapa_neuronas[v[0]][v[1]][1], self.mapa_neuronas[i_ganador][j_ganador][1]])
                self.axis[1].plot([v[0], i_ganador], [v[1], j_ganador])
                self.axis[1].plot(v[0], v[1], '*')
            
            ptos_x, ptos_y = self.get_pesos()
            self.axis[0].scatter(ptos_x, ptos_y)
            self.axis[0].plot(dato_random[0], dato_random[1], 's')
            self.axis[0].plot(self.mapa_neuronas[i_ganador][j_ganador][0], self.mapa_neuronas[i_ganador][j_ganador][1], '*')

            self.figure.canvas.draw()
            
            self.figure.canvas.flush_events()
            


            self.actualizar(dato_random, i_ganador, j_ganador)
            for i in range(0, self.f):
                for j in range(0, self.c):
                    if i==i_ganador and j==j_ganador:
                        continue
                    if self.esta_en_la_vecindad(radio, i_ganador, j_ganador, i, j):
                        self.actualizar(dato_random, i, j)
class kmeans:
    def __init__(self, path_datos, k, max_epocas=1000):
        self.X = pd.read_csv(path_datos, header=None).values
        self.X = self.X[:, :-3]
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
        idx = np.random.randint(0, len(self.X))
        return self.X[idx]
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
                # Para el dato 'd' computamos el centroide mas cercano
                dist_min = 999999
                c_mas_cercano = 0
                ind_c = 0
                
                #Prueba el patron con cada uno de los centroides
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
                if(len(self.S[ind_c]) != 0):
                    pos = pos / len(self.S[ind_c])
                else:
                    pos = pos/1
                self.ks[ind_c] = pos
                # Ahora le asignamos al centroide esta posicion
                #self.ks[ind_c] = pos
                ind_c += 1
            # Finalmente limpiamos los vectores en S
            if k != self.max_epocas-1:
                ind_c = 0
                for c_k in self.ks:
                    self.S[ind_c] = []
                    ind_c += 1
                
    def compactitud(self):
        self.compactitudes = []
        it = 0
        while it < len(self.ks):
            self.compactitudes.append(0)
            for i in range(0, len(self.S[it])):
                j = self.S[it][i]
                self.compactitudes[it] += float(np.sqrt((self.X[j] - self.ks[it]) @ (self.X[j] - self.ks[it])))
                
            if len(self.S[it]) == 0:
                self.compactitudes[it] = 0
            else:
                self.compactitudes[it] = self.compactitudes[it]/len(self.S[it])
                
            it += 1
            
        return self.compactitudes

p = kmeans('iris81_trn.csv', 3)
p.entrenar()
p.graficar()
compac = (p.compactitud())

com_tot = 0
for i in range(0, len(compac)):
    com_tot += compac[i]
print("Compactitud total del K-Means : ")
print(com_tot/len(compac))

som = SOM('iris81_trn.csv', 6, 6, 0.8)
som.entrenar()