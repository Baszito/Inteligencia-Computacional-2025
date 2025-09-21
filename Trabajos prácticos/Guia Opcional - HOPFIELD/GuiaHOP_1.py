import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sgn(v: float) -> float:
    if v == 0:
        return v
    else:
        if (v>0):
            return 1
        else:
            return -1
    
    
class Hopfield:

    def __init__(self, path_datos,dimensiones):
        self.X = pd.read_csv(path_datos, header=None).values #Memorias fundamentales
        self.dim = dimensiones #dimensiones de los patrones de entradas (filas x columnas)
        self.longitud_vector = dimensiones[0]*dimensiones[1]

        #Matriz de pesos de 5 x 5, diagonal principal 0, simetrica
        self.pesos = np.zeros((self.longitud_vector, self.longitud_vector))
        for i in range(0, self.pesos.shape[0]):
            for j in range(0, self.pesos.shape[1]):
                if i == j: 
                    continue
                else:
                    for k in range(0, self.X.shape[0]):
                        self.pesos[i][j] += self.X[k][j] * self.X[k][i]
                    self.pesos[i][j] /= self.longitud_vector

    

    def recuperacion(self,entrada):
        self.y= []
        self.y.append(entrada)
        for i in range(1, 100000):
            #Recuperacion
            self.y.append(np.copy(self.y[-1]))
            j_rnd = np.random.randint(low=0, high=self.longitud_vector)
            sum = 0
            for k in range(0, self.longitud_vector):
                sum += self.pesos[j_rnd][k] * self.y[i-1][k]
            if(sum == 0):
                self.y[i][j_rnd] = self.y[i-1][j_rnd]
            else:
                self.y[i][j_rnd] = sgn(sum)
        return self.y[99999]

                
        
hop = Hopfield('Trabajos prácticos/Guia Opcional - HOPFIELD/1.csv', [5, 5])
r = hop.recuperacion(np.array([-1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1]))
mat = r.reshape(5,5)   # convierte el vector en una matriz 5x5

for i in range(0, 5):
    for j in range(0, 5):
        color = 'red' if mat[i, j] == 1 else 'black'
        plt.plot(j, 4-i, 'o', color=color)  
plt.show()

