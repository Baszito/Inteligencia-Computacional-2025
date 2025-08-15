import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sgn import sgn

#Importar el csv
archivo = pd.read_csv('OR_trn.csv')
datos = archivo.to_numpy()
#Seleccion de parametros
gamma=0.01
max_epocas=10
criterio=0.05

#Inicializar pesos:
#Cantidad de entradas : 
cant_datos_por_fila = datos[1].size-1 #El -1 es porque la ultima es el resultado del or
yd = datos[:, cant_datos_por_fila]

ind = 0
cant_columnas=datos.shape[0]
print(cant_columnas)
w = [0.5, -0.5, 0.5]
unos=-1*np.ones([cant_columnas,1])
x = datos[:, 0:cant_datos_por_fila]
x = np.concatenate((unos, x), axis=1)
#x = datos[j, 0:cant_datos_por_fila]
#x = np.concatenate(([-1], x), axis=1)
#Entrenamiento

for i in range(0, max_epocas):
    tasa_error=0
    for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        w = w + (gamma/2)*(yd[j] - yn)*x[j]
        if(j%20==0):
            def x2(x1):
                return (w[0]/w[2]) -(w[1]/w[2])*x1
            #yes = np.zeros([cant_columnas, 1])
            xc = np.linspace(-1 , 1, 100)
            yes = []
            for k in xc:
                yes.append(x2(k))
            fig, ax = plt.subplots()  
            ax.grid(True) 
            ax.plot(xc, yes)  
            ax.plot([x[1],x[2]],'o')
            plt.show()  
            print(w)
            
    for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        if (yn!=yd[j]):
            tasa_error+=1
    tasa_error/=cant_columnas
    if(tasa_error<criterio):
        break
print(w)


