import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Capa import Capa

#Red es la clase que se encargara de manejar la informacion entre capas
class Red:
    def __init__(self, arquitectura):
        #Inicio armando la cantidad de capas segun el vector (arquitectura) que le paso
        #arquitectura = [cant_entradas, capa_o1, capa_o2, ..., cant_salidas]
        #La cantidad de capas será len(arquitectura) - 1
        self.capas = []
        for i in range(1, len(arquitectura)):
            #Se le pasa a cada capa la cantida de neuronas + la cantidad de neuronas de la capa anterior, que serian las entradas de la capa actual
            self.capas.append(Capa(arquitectura[i], arquitectura[i-1]))
        
        #errores sera un vector donde guardaremos los errores promedio de cada epoca en el entrenamiento
        self.errores = []

    #forward es el sistema de "avance" (propagacion hacia adelante)
    #Se calculan las salidas a partir de las entradas y los W de cada capa
    #El argumento es el patron actual con el que se esta trabajando
    def forward(self, x):
        #Agregamos los -1 correspondientes a los Bias DE LA ENTRADA
        salida = x

        #Para capa, hacemos el propio forward (calculamos las salidas a partir de las entradas)
        for capa in self.capas:
            #Insertamos el arreglo de forward a la salida de la capa y le agregamos un bias POR CADA CAPA OCULTA
            salida = capa.forward(salida)
            
        return salida

    #backward es lo que genera la retropropagacion
    #El argumento recibido representa la salida deseada del patron actual
    def backward(self, y_d):
       # y_d se espera columna (n_salidas, 1)
        error = y_d - self.capas[-1].output
        delta = self.capas[-1].backward(error)  #El primer delta de error se calcula haciendo backward de y_d - y_calculado
        
       #Propagar hacia atrás: se calculan los deltas de las demas capas teniendo en cuenta: el delta siguiente y la matriz W siguiente
        for i in reversed(range(len(self.capas)-1)):
            delta = self.capas[i].backward(delta, self.capas[i+1].W)
            
    #Actualizamos los pesos teniendo en cuenta el coeficiente de aprendizaje, los W actuales y los delta de error de cada capa
    def actualizar_pesos(self, u):
        for capa in self.capas:
            capa.actualizar_pesos(u)

    #entrenar se encarga de, justamente, hacer el proceso de entrenamiento de la red
    #Los argumentos son: el archivo de datos de entrenamiento, epocas maximas, coef de aprendizaje y la tolerancia
    def entrenar(self, archivo_csv, epocas=1000, u=0.01 , tolerancia=0.1):
        #En datos, se cargan todos los patrones de aprendizaje
        datos = pd.read_csv(archivo_csv, header=None).values
        
        #Separamos las entradas de las salidas deseadas
        X = datos[:, :-1]   # entradas
        Y = datos[:, -1]    # salida deseada
        
        #Asegurar que Y sea columna
        Y = Y.reshape(-1, 1)
        
        #Inicia el proceso de entrenamiento:
        #Por cada epoca...
        for epoca in range(epocas):
            error_total = 0
            #Por cada patron...
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

            #Calculamos el error promedio de la epoca y lo agregamos al vector de errores
            error_prom = error_total / len(X)
            self.errores.append(error_prom)
            
            #Cada 100 epocas, revisamos el error promedio
            if epoca % 100 == 0:
                print(f"Época {epoca}, Error: {error_prom:.6f}")
            
            #Si el error promedio no supera la tolerancia, el entrenamiento puede finalizar
            if error_prom < tolerancia:
                print("Criterio de aceptación alcanzado.")
                break

    #testear prueba la red neuronal con los datos de testeo del archivo
    #Argumentos: el archivo de datos de testeo
    def testear(self, archivo_csv):
        #PREGUNTAR EL JUEVES POR CRITERIOS DE ERROR EN TESTEO
        
        #datos contendra todos los datos leidos del archivo, y luego se separan entre entradas y salidas
        datos = pd.read_csv(archivo_csv, header=None).values
        X = datos[:, :-1]
        Y = datos[:, -1]
        er = 0
        aciertos = 0
        
        #Analizamos patron por patron
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
        
    #evolucionError muestra la grafica de la evolucion del error
    def evolucionError(self):
        #Eje de abscisas
        x_abs = list(range(1, len(self.errores)+1))
        
        plt.plot(x_abs, self.errores)
        plt.xlabel("Epocas")
        plt.ylabel("Error promedio")
        plt.title("Evolución del error")
        plt.grid(True)
        plt.show()
        
