import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#Cargamos los datos:
train_data = pd.read_csv("leukemia_train.csv")
test_data = pd.read_csv("leukemia_test.csv")

#Separamos los datos de entrenameinto y testeo
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

#Dimension de los patrones de entrada
num_caracteristicas = X_train.shape[1]

#Definimos la funcion de aptitud
def Aptitud(individuo):
    #seleccion: guarda los indices de las caracteristicas que tienen un 1 (variaran segun cada individuo de la poblacion)
    seleccion = np.where(individuo == 1)[0]
    if len(seleccion) == 0:
        #Si no se selecciono ninguna caracteristica, no hay aptitud que calcular
        return 0
    
    #El entrenamiento se hara con las columnas de las caracteristicas seleccionadas
    X_sel = X_train[:, seleccion]
    
    #Para la validacion cruzada, usaremos un SVC lineal
    clf = SVC(kernel = "linear")
    
    #Validacion cruzada
    aciertos = cross_val_score(clf, X_sel, y_train, cv=5)
    
    #Penalización para subconjuntos demasiado grandes
    #La idea sería encontrar el conjunto de caracteristicas que mejor resuelven, pero que no sean las 7129 que hay en total (una banda)
    penalizacion = 0.001 * (len(seleccion) / num_caracteristicas)
    return aciertos.mean() - penalizacion

#Operaciones geneticas
def cruzas_simples(p1, p2):
    #La mascara nos dira cual gen adquiere de p1 y cual de p2
    mascara = np.random.rand(len(p1)) < 0.5
    
    #El hijo hereda de p1 y p2 segun la mascara que declaramos
    hijo = np.where(mascara, p1, p2)
    return hijo

def mutaciones(individuo, tasa_de_mutacion = 0.01):
    #Para cada elemento del individuo, probamos si muta o no
    for i in range(len(individuo)):
        #Probamos con numeros en el rango [0.0, 1.0)
        if random.random() < tasa_de_mutacion:
            #Si el numero es menor que la tasa de mutacion, cambiamos el gen
            individuo[i] = 1 - individuo[i]
    return individuo

def seleccion_por_torneo(poblacion, aciertos, k = 3):
    #A las piñas muchachos
    #Seleccionamos k individuos al azar de la poblacion y su APTITUD
    seleccion = random.sample(list(zip(poblacion, aciertos)), k)
    
    #Ordenamos los seleccionados de mayor a menor
    #key = lambda x:x[1] -> Usa como parametro de comparacion el 2do elemento de seleccion (la aptitud del individuo)
    seleccion.sort(key=lambda x: x[1], reverse=True)
    
    #Retornamos al individuo de mayor aptitud (el que quedo 1ero en el ordenamiento pue)
    return seleccion[0][0]

#Algoritmo genetico
def algoritmo_genetico(tam_poblacion = 30, generaciones = 30, tasa_de_mutacion = 0.01):
    # población inicial (pocos genes activos al azar)
    poblacion = [np.random.choice([0,1], size = num_caracteristicas, p = [0.995, 0.005]) for _ in range(tam_poblacion)]

    #Recorremos las generaciones
    for g in range(generaciones):
        #Calculamos las aptitudes de la poblacion actual
        aciertos = [Aptitud(ind) for ind in poblacion]
        nueva_poblacion = []

        #Elitismo: Buscamos al individuo de mayor aptitud, y pasa directo a la siguiente generacion
        elite_idx = np.argmax(aciertos)
        ind_elite = poblacion[elite_idx]
        acierto_elite = aciertos[elite_idx]
        nueva_poblacion.append(ind_elite)

        #Reproduccion (53x0)
        #Hasta completar la nueva generacion, hacemos:
        while len(nueva_poblacion) < tam_poblacion:
            #Seleccionamos por competencia 2 individuos
            p1 = seleccion_por_torneo(poblacion, aciertos)
            p2 = seleccion_por_torneo(poblacion, aciertos)
            
            #El hijo sera el resultado de la cruza simple de los 2 progenitores seleccionados
            hijo = cruzas_simples(p1, p2)
            
            #Mutamos al hijo para explorar zonas de solucion nuevas (o no, la componente aleatoria siempre esta)
            hijo = mutaciones(hijo, tasa_de_mutacion)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion
        print(f"Gen {g+1}/{generaciones} - Mejor fitness: {acierto_elite:.4f}")

    #Devolvemos el mejor individuo encontrado en la ultima generacion
    aciertos_finales = [Aptitud(ind) for ind in poblacion]
    best_idx = np.argmax(aciertos_finales)
    return poblacion[elite_idx]

#Ejecutamos el algoritmo, y vemos cuantas caracteristicas quedaron seleccionadas
mejor_IND = algoritmo_genetico(tam_poblacion = 40, generaciones = 50, tasa_de_mutacion = 0.02)
caract_seleccion = np.where(mejor_IND == 1)[0]
print(f"Genes seleccionados: {len(caract_seleccion)}")

#Entrenamiento final con el individuo ganador
clf = SVC(kernel="linear")
clf.fit(X_train[:, caract_seleccion], y_train)
y_pred = clf.predict(X_test[:, caract_seleccion])
acc = accuracy_score(y_test, y_pred)
print(f"Exactitud en test: {acc:.4f}")
