import numpy as np
import pandas as pd

#Funcion de activacion
def sigmoid(x):
    return (2/(1 + np.exp(-x))) -1

#La sigmoide 1/(1 + exp(-x)) toma valores demaciado altos, tendiendolos al infinito y generando overflow

def d_sigmoid(x):
    return (1/2)*(1+x)*(1-x)

class Capa:
    def __init__(self, neuronas, entradas):
        #para inicializar la capa, se colocan pesos random entre -0.5 y 0.5
        #Se le pasa a cada capa la cantidad de entradas y de neuronas
        self.W = np.random.randn(neuronas, entradas+1) - 0.5*np.ones((neuronas, entradas+1))
        self.counter = 0
        self.neuronas = neuronas
        self.entradas = entradas
        self.input = None
        self.output = None
        self.delta = None
        
    #Calculamos las salidas de la capa    
    def forward(self, x):
        x = np.insert(x, 0, -1)
        self.input = x
        z = np.matmul(self.W, x.T)
        self.output = sigmoid(z)
        return self.output
    
    #Calculamos los delta de gradiente del error
    def backward(self, delta_siguiente, W_siguiente = None):
        #print("OUTPUT: ")
        #print(self.output)
        #print("DELTA SIGUIENTE: ")
        #print(delta_siguiente)
        if W_siguiente is None:
            self.delta = np.multiply(delta_siguiente, d_sigmoid(self.output))
        else:

            self.delta = np.multiply(np.matmul(W_siguiente[:,1:].T, delta_siguiente).reshape(-1, 1), d_sigmoid(self.output).reshape(-1, 1))

        return self.delta
   
    #actualizamos los pesos en relacion a los pesos actuales, los delta y el coeficiente de aprendizaje
    def actualizar_pesos(self, u):
        #dW = u*np.dot(self.delta, self.input)
        dW = self.W.copy()
        self.delta = self.delta.flatten()
        self.input = self.input.flatten()
        
        for i in range(0, dW.shape[0]):
            for j in range(0, dW.shape[1]):
                dW[i, j] = u * self.delta[i] * self.input[j]
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i, j] =  self.W[i, j] + dW[i, j]
                #
        if self.counter % 10 == 0:
            pass
        self.counter+=1
