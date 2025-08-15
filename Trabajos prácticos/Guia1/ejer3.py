import pandas as pd
import numpy as np
from sgn import sgn

#-------------------------------OR ORIGINAL-------------------------------#
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
    for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        if (yn!=yd[j]):
            tasa_error+=1
    tasa_error/=cant_columnas
    if(tasa_error<criterio):
        break
print("Pesos para el OR original")
print(w)

#Test
archivo_test=pd.read_csv("OR_tst.csv")
datos_test = archivo_test.to_numpy()
cant_datos_por_fila = datos_test[1].size-1
yd_test=datos_test[:, cant_datos_por_fila]
cant_columnas=datos_test.shape[0]
tasa_error=0
unos=-1*np.ones([cant_columnas,1])
x = datos_test[:, 0:cant_datos_por_fila]
x = np.concatenate((unos, x), axis=1)
for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        if (yn!=yd_test[j]):
            tasa_error+=1
        tasa_error/=cant_columnas
print(tasa_error)
#-------------------------------OR ORIGINAL-------------------------------#
#Importar el csv
archivo = pd.read_csv('OR_50_trn.csv')
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
    for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        if (yn!=yd[j]):
            tasa_error+=1
    tasa_error/=cant_columnas
    if(tasa_error<criterio):
        break
print("Pesos para el OR 50")
print(w)

#Test
archivo_test=pd.read_csv("OR_50_tst.csv")
datos_test = archivo_test.to_numpy()
cant_datos_por_fila = datos_test[1].size-1
yd_test=datos_test[:, cant_datos_por_fila]
cant_columnas=datos_test.shape[0]
tasa_error=0
unos=-1*np.ones([cant_columnas,1])
x = datos_test[:, 0:cant_datos_por_fila]
x = np.concatenate((unos, x), axis=1)
for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        if (yn!=yd_test[j]):
            tasa_error+=1
        tasa_error/=cant_columnas
print(tasa_error)

#-------------------------------OR90 ORIGINAL-------------------------------#
#Importar el csv
archivo = pd.read_csv('OR_90_trn.csv')
datos = archivo.to_numpy()
#Seleccion de parametros
gamma=0.01
max_epocas=1
criterio=0.0000000000000000000000001

#Inicializar pesos:
#Cantidad de entradas : 
cant_datos_por_fila = datos[1].size-1 #El -1 es porque la ultima es el resultado del or
yd = datos[:, cant_datos_por_fila]

ind = 0
cant_columnas=datos.shape[0]
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
    for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        if (yn!=yd[j]):
            tasa_error+=1
    tasa_error/=cant_columnas
    if(tasa_error<criterio):
        break
print("Pesos para el OR 90")
print(w)

#Test
archivo_test=pd.read_csv("OR_90_tst.csv")
datos_test = archivo_test.to_numpy()
cant_datos_por_fila = datos_test[1].size-1
yd_test=datos_test[:, cant_datos_por_fila]
cant_columnas=datos_test.shape[0]
tasa_error=0
unos=-1*np.ones([cant_columnas,1])
x = datos_test[:, 0:cant_datos_por_fila]
x = np.concatenate((unos, x), axis=1)
for j in range(0, cant_columnas):
        yn = sgn(np.dot(w, x[j]))
        if (yn!=yd_test[j]):
            tasa_error+=1
        tasa_error/=cant_columnas
print(tasa_error)

