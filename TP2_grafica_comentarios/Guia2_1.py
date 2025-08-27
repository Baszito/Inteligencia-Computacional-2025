from Red import Red
import matplotlib.pyplot as plt

# red[0] = entradas; red[1 y 2] = numero de neuronas por capa
red = Red([2, 2, 1])

#Entrenamos la red con el metodo "entrenar" que recibe:   El archivo de datos de entrenamiento;
#                                                         Cantidad maxima de epocas;
#                                                         Coeficiente de aprendizaje;
#                                                         Criterio de aceptacion;
red.entrenar("XOR_trn.csv", 1000, 0.01, 0.005)

#Ahora testeamos la red con el metodo "testear", enviandole el archivo con los datos de testeo
red.testear("XOR_tst.csv")

#Podemos graficar la evolucion del error por epoca con la siguiente funcion:
red.evolucionError()