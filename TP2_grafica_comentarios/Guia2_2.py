from Red import Red
import matplotlib.pyplot as plt

# El criterio de la estructura más apropiada es de carácter ambiguo.

# Se explorarán diferentes soluciones.
red = Red([2, 8, 1])

#Entrenamos la red con el metodo "entrenar" que recibe:   El archivo de datos de entrenamiento;
#                                                         Cantidad maxima de epocas;
#                                                         Coeficiente de aprendizaje;
#                                                         Criterio de aceptacion;
red.entrenar("TP2_grafica_comentarios/concent_trn.csv", 1000, 0.01, 0.95)

#Ahora testeamos la red con el metodo "testear", enviandole el archivo con los datos de testeo
red.testear("TP2_grafica_comentarios/concent_tst.csv")

#Podemos graficar la evolucion del error por epoca con la siguiente funcion:
red.evolucionError()