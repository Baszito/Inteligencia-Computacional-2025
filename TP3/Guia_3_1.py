from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Generamos los datos del dataset, y los separamos en entradas y salidas deseadas
X, y = load_digits(return_X_y = True)

#Partición: El 80% de los datos iran al entrenamiento, el 20% restante seran nuestros datos de testeo
#Shuffle = True hace que los patrones generados se mezclen, evitando que podamos agarrar patrones con la misma salida deseada
#Hay otro parametro que se puede agregar llamado "random_state", que es una semilla que nos permite controlar la aleatoriedad de los resultados
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

#hidden_layer_sizes = (_, _) funciona como "arquitectura" en la guia 2. Es un arreglo, donde cada elemento es la cant de neuronas de cada capa
#                                                                                       y la cantidad de elementos es la cantidad de capas ocultas
#Cantidad de epocas maximas = 1000
clf = MLPClassifier(hidden_layer_sizes=(3, 1), max_iter=1000)
#Funcion de entrenamiento del perceptron
clf.fit(X_trn, y_trn)
#Calculamos las salidas del perceptron en base a los datos de testeo
y_calculado = clf.predict(X_tst)

#Presicion (tasa de aciertos) del perceptron
presicion_P_unica = accuracy_score(y_tst, y_calculado)
print("Precisión con una unica particion:", presicion_P_unica)

#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#

#Por comodidad (y no repetir codigo al pedo), el entrenamiento y testeo de cada KFlod lo hacemos en una funcion
def evaluar_con_KFold(X, y, k):
    #k es el numero de particiones que vamos a generar (Se divide la totalidad de los datos en k partes)
    #Shuffle = True hace que los patrones generados se mezclen, evitando que podamos agarrar patrones con la misma salida deseada
    #Hay otro parametro que se puede agregar llamado "random_state", que es una semilla que nos permite controlar la aleatoriedad de los resultados
    kf = KFold(n_splits=k, shuffle=True)
    
    #Vector de tasas de aciertos para cada particion generada
    tasas_de_acierto = []

    #El bucle recorre las particiones, obteniendo X_trn, X_tst, y_trn e y_tst segun corresponda a cada particion
    #kf.split(X), retorna las tuplas de indices correspondientes a cada particion, separando indices de entrenamiento y testeo
    for trn_i, tst_i in kf.split(X):
        X_trn, X_tst = X[trn_i], X[tst_i]
        y_trn, y_tst = y[trn_i], y[tst_i]

        #hidden_layer_sizes = (_, _) funciona como "arquitectura" en la guia 2. Es un arreglo, donde cada elemento es la cant de neuronas de cada capa
                                                                                #y la cantidad de elementos es la cantidad de capas ocultas
        #Cantidad de epocas maximas = 1000
        clf = MLPClassifier(hidden_layer_sizes=(3, 1), max_iter=1000)
        #Entrenamos el perceptron
        clf.fit(X_trn, y_trn)
        #Calculamos la salida en base al perceptron armado
        y_calculado = clf.predict(X_tst)
        
        #Agregamos la tasa de aciertos de la particion a la lista de tasas
        tasas_de_acierto.append(accuracy_score(y_tst, y_calculado))

    #Retornamos la media y la varianza de las tasas de aciertos para el analisis
    return np.mean(tasas_de_acierto), np.var(tasas_de_acierto)

# KFold con 5 particiones
media_5, varianza_5 = evaluar_con_KFold(X, y, 5)
print(f"KFold(5) - media: {media_5:.4f}, varianza: {varianza_5:.6f}")

# KFold con 10 particiones
media_10, varianza_10 = evaluar_con_KFold(X, y, 10)
print(f"KFold(10) - media: {media_10:.4f}, varianza: {varianza_10:.6f}")
