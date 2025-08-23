import numpy as np
import pandas as pd
from Capa import Capa


class Red:
    def __init__(self, arquitectura):
        self.capas = []
        for i in range(1, len(arquitectura)):
            self.capas.append(Capa(arquitectura[i], arquitectura[i-1]))

    def forward(self, x):
        salida = np.concatenate([-1*np.ones((len(x), 1)), x], axis=1)
        for capa in self.capas:
            salida = np.insert(capa.forward(salida), 0, -1)
            
        return salida

    def backward(self, y_d):
        # y_d se espera columna (n_salidas, 1)
        error = self.capas[-1].output - y_d
        delta = self.capas[-1].backward(error)  # capa de salida

        # Propagar hacia atrás
        for i in reversed(range(len(self.capas)-1)):
            delta = self.capas[i].backward(delta, self.capas[i+1].W)

    def actualizar_pesos(self, u):
        for capa in self.capas:
            capa.actualizar_pesos(u)

    def entrenar(self, archivo_csv, epocas=1000, u=0.1, tolerancia=0.01):
        datos = pd.read_csv(archivo_csv, header=None).values
        X = datos[:, :-1]   # entradas
        Y = datos[:, -1]    # salida deseada
        
        # asegurar que Y sea columna
        Y = Y.reshape(-1, 1)
        
        for epoca in range(epocas):
            error_total = 0
            for i in range(len(X)):
                
                # Forward
                y_pred = self.forward(X[i].reshape(1, -1))

                # Error cuadrático
                error_total += np.mean((Y - y_pred) ** 2)

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
        datos = pd.read_csv(archivo_csv, header=None).values
        X = datos[:, :-1]
        Y = datos[:, -1]
        ones = -1*np.ones((datos.shape[0], 1))  
        X = np.concatenate([ones, X], axis=1)

        aciertos = 0
        for i in range(len(X)):
            x = X[i].reshape(-1, 1)
            y = Y[i]
            y_pred = self.forward(x)
            
            if y_pred == y:
                aciertos += 1
        
        ratio = aciertos / len(X)
        print(f"Ratio de aprobación: {ratio*100:.2f}%")
        return ratio