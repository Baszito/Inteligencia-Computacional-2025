from sklearn.datasets import load_wine

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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

X, Y = load_wine(return_X_y=True)

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
        if que_metodo == "BAGGING_SVC":
            clf = BaggingClassifier(estimator=SVC(), n_estimators=20)
        if que_metodo == "BAGGING":
            
            clf = BaggingClassifier(estimator=LinearDiscriminantAnalysis(), n_estimators=20)
        if que_metodo == "ADABOOST":
            clf = AdaBoostClassifier(n_estimators=20)
        #Entrenamos el perceptron
        clf.fit(X_trn, y_trn)
        #Calculamos la salida en base al perceptron armado
        y_calculado = clf.predict(X_tst)
        
        #Agregamos la tasa de aciertos de la particion a la lista de tasas
        tasas_de_acierto.append(accuracy_score(y_tst, y_calculado))
    #Retornamos la media y la varianza de las tasas de aciertos para el analisis
    return np.mean(tasas_de_acierto), np.var(tasas_de_acierto)

media_5_bagging_svc, varianza_5_bagging_svc = evaluar_con_KFold(X, Y, 5, (64, 32), "BAGGING_SVC")


# DEJAMOS EL ADL PORQUE PROBANDO FUE EL QUE MEJOR ANDUVO
media_5_bagging, varianza_5_bagging = evaluar_con_KFold(X, Y, 5, (64, 32), "BAGGING")
media_5_adaboost, varianza_5_adaboost = evaluar_con_KFold(X, Y, 5, (64, 32), "ADABOOST")


print("MEDIA DE BAGGING CON SVC (5 PARTICIONES): ", media_5_bagging_svc)
print("VARIANZA DE BAGGING CON SVC (5 PARTICIONES): ", varianza_5_bagging_svc)

print("MEDIA DE BAGGING CON ANÁLISIS DISCRIMANTE LINEAL (5 PARTICIONES): ", media_5_bagging)
print("VARIANZA DE BAGGING CON ANÁLISIS DISCRIMANTE LINEAL (5 PARTICIONES): ", varianza_5_bagging)

print("MEDIA DE ADABOOST (5 PARTICIONES): ", media_5_adaboost)
print("VARIANZA DE ADABOOST (5 PARTICIONES): ", varianza_5_adaboost)


