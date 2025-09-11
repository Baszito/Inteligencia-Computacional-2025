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
import matplotlib.pyplot as plt

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
        if que_metodo == "BAGGING_DTC":
            clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=20)
        if que_metodo == "BAGGING_LDA":
            clf = BaggingClassifier(estimator=LinearDiscriminantAnalysis(), n_estimators=20)
        if que_metodo == "ADABOOST_DTC":
            clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=20)
        if que_metodo == "ADABOOST_SVC":
            clf = AdaBoostClassifier(estimator=SVC(),n_estimators=20)
        #Entrenamos el perceptron
        clf.fit(X_trn, y_trn)
        #Calculamos la salida en base al perceptron armado
        y_calculado = clf.predict(X_tst)
        
        #Agregamos la tasa de aciertos de la particion a la lista de tasas
        tasas_de_acierto.append(accuracy_score(y_tst, y_calculado))
    #Retornamos la media y la varianza de las tasas de aciertos para el analisis
    return np.mean(tasas_de_acierto), np.var(tasas_de_acierto)

media_5_bagging_dtc, varianza_5_bagging_dtc = evaluar_con_KFold(X, Y, 5, (64, 32), "BAGGING_DTC")
media_5_bagging_lda, varianza_5_bagging_lda = evaluar_con_KFold(X, Y, 5, (64, 32), "BAGGING_LDA")
media_5_bagging_svc, varianza_5_bagging_svc = evaluar_con_KFold(X, Y, 5, (64, 32), "BAGGING_SVC")
media_5_adaboost_dtc, varianza_5_adaboost_dtc = evaluar_con_KFold(X, Y, 5, (64, 32), "ADABOOST_DTC")
media_5_adaboost_svc, varianza_5_adaboost_svc = evaluar_con_KFold(X, Y, 5, (64, 32), "ADABOOST_SVC")



print("MEDIA DE BAGGING CON DTC (5 PARTICIONES): ", media_5_bagging_dtc)
print("VARIANZA DE BAGGING CON DTC (5 PARTICIONES): ", varianza_5_bagging_dtc)
###En multiples pruebas, LDA nos dio el mejor resultado con bagging
###lamentablemente no es compatible con adaboost, por un tema de los pesos en las librerias
print("MEDIA DE BAGGING CON LDA (5 PARTICIONES): ", media_5_bagging_lda)
print("VARIANZA DE BAGGING CON LDA (5 PARTICIONES): ", varianza_5_bagging_lda)
print("MEDIA DE BAGGING CON SVC (5 PARTICIONES): ", media_5_bagging_svc)
print("VARIANZA DE BAGGING CON SVC (5 PARTICIONES): ", varianza_5_bagging_svc)

print("MEDIA DE ADABOOST CON DTC (5 PARTICIONES): ", media_5_adaboost_dtc)
print("VARIANZA DE ADABOOST CON DTC (5 PARTICIONES): ", varianza_5_adaboost_dtc)
print("MEDIA DE ADABOOST CON SVC (5 PARTICIONES): ", media_5_adaboost_svc)
print("VARIANZA DE ADABOOST CON SVC (5 PARTICIONES): ", varianza_5_adaboost_svc)

plt.style.use('_mpl-gallery')

# plot
fig, ax = plt.subplots()
x = np.arange(5)
y=[media_5_bagging_dtc,media_5_bagging_lda,media_5_bagging_svc,media_5_adaboost_dtc,media_5_adaboost_svc]
ax.bar(x, y, width=0.8, edgecolor="white", linewidth=0.7)
labels = ["Bag_DTC","Bag_LDA", "Bag_SVC","Ada_DTC", "Ada_SVC"]
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set(xlim=(-0.5, 5.5), xticks=np.arange(5),
       ylim=(0, 1), yticks=np.linspace(0,1,30))
ax.set_label("Medias")
plt.show()

# plot
fig, ax = plt.subplots()
x = np.arange(5)
y=[varianza_5_bagging_dtc,varianza_5_bagging_lda,varianza_5_bagging_svc,varianza_5_adaboost_dtc,varianza_5_adaboost_svc]
ax.bar(x, y, width=0.8, edgecolor="white", linewidth=0.7)
labels = ["Bag_DTC","Bag_LDA", "Bag_SVC","Ada_DTC", "Ada_SVC"]
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set(xlim=(-0.5, 5.5), xticks=np.arange(5),
       ylim=(0, 0.02), yticks=np.linspace(0,0.02,30))
ax.set_label("Medias")
plt.show()