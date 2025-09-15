import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
plt.ion()
class SOM:
    def __init__(self, path_datos, nro_filas_neuronas, nro_columnas_neuronas, velocidad_entrenamiento, velocidad_entrenamiento_final = 0.01, max_epocas=1000):
        self.X = pd.read_csv(path_datos, header=None).values
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
            


            

# Circulo
#s = SOM('TP4/circulo.csv', 10, 10, 0.8)
#s.entrenar()

# T 
s = SOM('TP4/te.csv', 100, 1, 0.8)
s.entrenar()
plt.ioff()
plt.show()