import numpy as np
import pandas as pd

#Funcion de activacion
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return (1/2)*(1+x)*(1-x)

class Capa:
    def __init__(self, neuronas, entradas):
        self.W = np.random.randn(neuronas, entradas+1) - 0.5*np.ones((neuronas, entradas+1))
        
        self.neuronas = neuronas
        self.entradas = entradas
        self.input = None
        self.output = None
        self.delta = None
        
    #Calculamos las salidas de la capa    
    def forward(self, x):
        self.input = x
        z = np.dot(x, self.W.T)
        self.output = sigmoid(z.T)
        return self.output
    
    #Calculamos los delta de gradiente del error
    def backward(self, delta_siguiente, W_siguiente = None):
        if W_siguiente is None:
            self.delta = delta_siguiente*d_sigmoid(self.output)
        else:
            print(W_siguiente[:,1:].T)
            self.delta = np.dot(np.dot(W_siguiente[:,1:].T, delta_siguiente), d_sigmoid(self.output))
            
        return self.delta
    
    #actualizamos los pesos en relacion a los pesos actuales, los delta y el coeficiente de aprendizaje
    def actualizar_pesos(self, u):
        dW = u*np.dot(self.delta, self.input)
        dw_mat = np.dot(dW, np.ones((self.neuronas, self.entradas)))
        self.W = self.W + dW