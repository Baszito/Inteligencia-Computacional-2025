import numpy as np
import pandas as pd
from Capa import Capa
#hola valen que ondab


class Red:
    def __init__(self, arquitectura):
        #Inicio armando la cantidad de capas segun la arquitectura que le paso
        self.capas = []
        for i in range(1, len(arquitectura)):
            #Se le pasa a cada capa la cantida de neuronas + la cantidad de neuronas de la capa anterior, que serian las entradas de la capa actual
            self.capas.append(Capa(arquitectura[i], arquitectura[i-1]))

    def forward(self, x):
        #Propagacion hacia adelante. 
        #Agregamos los -1 correspondientes a los Bias DE LA ENTRADA
        salida = x

        for capa in self.capas:
            #Insertamos el arreglo de forward a la salida de la capa y le agregamos un bias POR CADA CAPA OCULTA
            salida = capa.forward(salida)
            
        return salida

    def backward(self, y_d):
       # y_d se espera columna (n_salidas, 1)
        error = y_d - self.capas[-1].output
        delta = self.capas[-1].backward(error)  # capa de salida
       # Propagar hacia atrás
        for i in reversed(range(len(self.capas)-1)):
            delta = self.capas[i].backward(delta, self.capas[i+1].W)
    def actualizar_pesos(self, u):
        for capa in self.capas:
            capa.actualizar_pesos(u)

    def entrenar(self, archivo_csv, epocas=1000, u=0.01 , tolerancia=0.1):
        datos = pd.read_csv(archivo_csv, header=None).values
        X = datos[:, :-1]   # entradas
        Y = datos[:, -1]    # salida deseada
        
        # asegurar que Y sea columna
        Y = Y.reshape(-1, 1)
        
        for epoca in range(epocas):
            error_total = 0
            for i in range(len(X)):

                # Forward
                xTemp = X[i]
                y_pred = self.forward(xTemp.reshape(1, -1))
                # Error cuadrático instantaneo
                error_total += (1/2)*np.mean((Y[i] - y_pred) ** 2)

                # Backward
                self.backward(Y[i])

                # Actualizar pesos
                self.actualizar_pesos(u)

            error_prom = error_total / len(X)
            if epoca % 100 == 0:
                print(f"Época {epoca}, Error: {error_prom:.6f}")
            
            if error_prom < tolerancia:
                print("Criterio de aceptación alcanzado.")
                break

    def testear(self, archivo_csv):
        #PREGUNTAR EL JUEVES POR CRITERIOS DE ERROR EN TESTEO
        datos = pd.read_csv(archivo_csv, header=None).values
        X = datos[:, :-1]
        Y = datos[:, -1]
        er = 0
        aciertos = 0
        for i in range(len(X)):
            x = X[i].reshape(-1, 1)
            y = Y[i]
            y_pred = self.forward(x)
            er += (y - y_pred)**2
            #if y_pred == y:
            #   aciertos += 1
        er /= 2
        er_avg = er / len(X)
        #ratio = aciertos / len(X)
        #print(f"Ratio de aprobación: {ratio*100:.2f}%")
        print("El error promedio es: ", er_avg)
        #return ratio
