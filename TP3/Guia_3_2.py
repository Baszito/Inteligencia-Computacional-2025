from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

def evaluar_con_KFold(X, y, k, Hidden_layer_sizes, que_metodo):
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
        clf = None
        if que_metodo == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=Hidden_layer_sizes, max_iter=10000)
        if que_metodo == "GNB":
            clf = GaussianNB()
        if que_metodo == "NN":
            clf = KNeighborsClassifier(n_neighbors=3)
        if que_metodo == "LDA":
            clf = LinearDiscriminantAnalysis()
        if que_metodo == "DTC":
            clf = DecisionTreeClassifier()
        if que_metodo == "SVC":
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        #Entrenamos el perceptron
        clf.fit(X_trn, y_trn)
        #Calculamos la salida en base al perceptron armado
        y_calculado = clf.predict(X_tst)
        
        #Agregamos la tasa de aciertos de la particion a la lista de tasas
        tasas_de_acierto.append(accuracy_score(y_tst, y_calculado))

    #Retornamos la media y la varianza de las tasas de aciertos para el analisis
    return np.mean(tasas_de_acierto), np.var(tasas_de_acierto)

X, y = load_digits(return_X_y = True)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

media_5_mlp, varianza_5_mlp = evaluar_con_KFold(X, y, 5, (64, 32), "MLP")
media_5_gnb, varianza_5_gnb = evaluar_con_KFold(X, y, 5, (64, 32), "GNB")
media_5_lda, varianza_5_lda = evaluar_con_KFold(X, y, 5, (64, 32), "LDA")
media_5_nn, varianza_5_nn = evaluar_con_KFold(X, y, 5, (64, 32), "NN")
media_5_dtc, varianza_5_dtc = evaluar_con_KFold(X, y, 5, (64, 32), "DTC")
media_5_svc, varianza_5_svc = evaluar_con_KFold(X, y, 5, (64, 32), "SVC")

print("Precisión media con MLP y 5 particiones:", media_5_mlp)
print("Varianza media con MLP y 5 particiones:", varianza_5_mlp)

print("Precisión media con Naive Bayes (Gaussiano) y 5 particiones:", media_5_gnb)
print("Varianza media con Naive Bayes (Gaussiano) y 5 particiones:", varianza_5_gnb)

print("Precisión media con análisis discrimante lineal y 5 particiones:", media_5_lda)
print("Varianza media con análisis discriminante lineal y 5 particiones:", varianza_5_lda)

print("Precisión media con Nearest Neighbours y 5 particiones:", media_5_nn)
print("Varianza media con Nearest Neighbours y 5 particiones:", varianza_5_nn)

print("Precisión media con árboles de decisión y 5 particiones:", media_5_dtc)
print("Varianza media con árboles de decisión y 5 particiones:", varianza_5_dtc)

print("Precisión media con máquinas de soporte vectorial y 5 particiones:", media_5_svc)
print("Varianza media con máquinas de soporte vectorial y 5 particiones:", varianza_5_svc)




