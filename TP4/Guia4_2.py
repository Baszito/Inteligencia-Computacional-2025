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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import time

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
            print(c_k)
            msize += 3
        plt.show()
    def get_color(self, i):
        return (i/len(self.ks), 0, 1 - i/len(self.ks))
    # Devuelve un índice al centroide ganador
    def get_centroide_ganador(self, i):
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
        return c_mas_cercano
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
                
    def compactitud(self) -> np.ndarray:
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
    
class SOM:
    def __init__(self, path_datos, nro_filas_neuronas, nro_columnas_neuronas, velocidad_entrenamiento, velocidad_entrenamiento_final = 0.01, max_epocas=1000):
        self.X = pd.read_csv(path_datos, header=None).values
        self.Y = self.X[:, 4:7]
        self.X = self.X[:, :-3]
        self.velocidad_entrenamiento = velocidad_entrenamiento
        self.velocidad_entrenamiento_inicial = velocidad_entrenamiento
        self.velocidad_entrenamiento_final = velocidad_entrenamiento_final
        self.max_epocas = max_epocas

        self.patronesxN = {}
        self.clasificacion = np.zeros(( nro_columnas_neuronas*nro_filas_neuronas,1))
        for i in range(0, nro_columnas_neuronas*nro_filas_neuronas):
            self.patronesxN[i] = []

        self.mapa_neuronas = []
        self.mapa_neuronas_ganadoras = []
        self.f = nro_filas_neuronas
        self.c = nro_columnas_neuronas
        for i in range(0, nro_filas_neuronas):
            self.mapa_neuronas.append([])
            self.mapa_neuronas_ganadoras.append([0])
            for j in range(0, nro_columnas_neuronas):
                self.mapa_neuronas[-1].append(self.generar_vector())
                self.mapa_neuronas_ganadoras[-1].append(0)

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
    def get_color(self, i, j):
        return (1-i/self.f, j/self.c, 0)
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

            dato_random_i = np.random.randint(self.X.shape[0])
            dato_random = self.X[dato_random_i, :]
            [i_ganador, j_ganador] = self.neurona_ganadora(dato_random)


            self.actualizar(dato_random, i_ganador, j_ganador)
            for i in range(0, self.f):
                for j in range(0, self.c):
                    if i==i_ganador and j==j_ganador:
                        continue
                    if self.esta_en_la_vecindad(radio, i_ganador, j_ganador, i, j):
                        self.actualizar(dato_random, i, j)
        self.actualizar_matriz_ganadoras()
        
    def actualizar_matriz_ganadoras(self):
        for i in range(0, self.X.shape[0]):
            [i_g, j_g] = self.neurona_ganadora(self.X[i, :])
            self.mapa_neuronas_ganadoras[i_g][j_g] += 1
    def val_max_matriz_ganadora(self):
        m = -99999999999
        for i in range(0, self.f):
            for j in range(0, self.c):
                if (self.mapa_neuronas_ganadoras[i][j] > m):
                    m = self.mapa_neuronas_ganadoras[i][j]
        return m
    def pertenencias(self):
            
            for i in range(0, self.X.shape[0]):
                patron_actual = self.X[i]
                [i_ganador, j_ganador] = self.neurona_ganadora(patron_actual)
                indice = i_ganador*self.f + j_ganador
                self.patronesxN[indice].append(i)
            return self.patronesxN
        
    def clasificacion_Iris(self):
        self.pertenencias()
        for i in range(0, self.X.shape[0]):
            for j in range(0, len(self.patronesxN)):
                for k in range(0, len(self.patronesxN[j])):
                    if self.patronesxN[j][k] == i:
                        if (self.Y[i, :] == np.array([-1, -1, 1])).all():
                            #setosa
                            self.clasificacion[j] = 1
                        else:
                            if (self.Y[i, :] == np.array([-1, 1, -1])).all():
                                #versicolor
                                self.clasificacion[j] = 2
                            else:
                                #virginica
                                self.clasificacion[j] = 3
                         
        return self.clasificacion
        

def graficar_SOM_y_KMEANS(som: SOM, kmeans: kmeans, dim_x=0, dim_y=1):
    
    fig, ax = plt.subplots(4)
    fig.subplots_adjust(hspace=0.5)
    X = som.X
    for i in range(0, X.shape[0]):
        [i_ganador_som, j_ganador_som] = som.neurona_ganadora(X[i, :])
        centroide_ganador_kmeans = kmeans.get_centroide_ganador(i)
        
        ax[0].plot([X[i][dim_x]], [X[i][dim_y]], color=som.get_color(i_ganador_som, j_ganador_som), marker='o')
        ax[0].set_title("SOM") #En el som los colores van de rojo a amarillo, dependiendo de su ganador en i y en j(coordenadas de las neuronas)
        ax[1].plot([X[i][dim_x]], [X[i][dim_y]], color=kmeans.get_color(centroide_ganador_kmeans), marker='o')
        ax[1].set_title("K-MEANS")#En el K-means los colores van de azul a rojo, dependiendo de su ganador en i y en j(coordenadas de las neuronas)
        ax[2].set_title("SOM-Mapa topologico (Rojo: Menos frecuencia, Verde: Mas frecuencia)") #En el som los colores van de rojo a amarillo, dependiendo de su ganador en i y en j(coordenadas de las neuronas)
    
    for i in range(0, som.f):
        for j in range(0, som.c):
            _color = (1 - som.mapa_neuronas_ganadoras[i][j]/som.val_max_matriz_ganadora(), som.mapa_neuronas_ganadoras[i][j]/som.val_max_matriz_ganadora() ,0)
            ax[2].plot(i, j, marker='o', color=_color)
    clasificacion = som.clasificacion_Iris()
    print('CLASIFICACION: ', clasificacion)
    for i in range(0, som.f):
        
        for j in range(0, som.c):
            _color = None
            
            if (clasificacion[i*som.f + j] == 1):
                # Cetácea, verde
                _color = (0, 1, 0)
            else:
                if (clasificacion[i*som.f + j] == 2):
                # Versicolor, rojo
                    _color = (1, 0, 0)
                else:
                    if(clasificacion[i*som.f + j] == 3):
                    # Virginica, azul
                        _color = (0, 0, 1)
                    else:
                        _color = (0, 0, 0)
                
            ax[3].plot(i, j, marker='o', color=_color)
    ax[3].set_title("Clasificacion de neuronas con SOM (Verde - Setosa / Rojo - Versicolor / Azul - Versicolor)")
    plt.show()

p = kmeans('TP4\iris81_trn.csv', 5)
p.entrenar()
compac = (p.compactitud())

com_tot = 0
for i in range(0, len(compac)):
    com_tot += compac[i]
print("Compactitud total del K-Means : ")
print(com_tot)

s = SOM('TP4\iris81_trn.csv', 5, 5, 0.8)
s.entrenar()

graficar_SOM_y_KMEANS(s, p, 0, 1)


#Matriz de Contingencia:
##Las filas representan los elementos de k_means (1 por cada k)
##Las columnas representan los elementos del SOM (1 por cada neurona)
##Los elementos consisten en la cantidad de patrones coincidentes de cada elemento
##Es decir, cada elemento es la cantidad de patrones que coinciden en la clase de la fila y columna
##Por ejemplo, el elemento(1, 1) es la cantidad de coincidencias entre la clase 1 de k_means y la clase 1 de SOM
Matriz_de_contengencia(p.S, s.pertenencias())