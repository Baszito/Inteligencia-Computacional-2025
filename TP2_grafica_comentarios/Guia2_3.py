from Red import Red
import matplotlib.pyplot as plt

# El criterio de la estructura más apropiada es de carácter ambiguo.

# Se explorarán diferentes soluciones.
red = Red([4, 3, 3, 3])

red_1 = Red([4, 7, 7, 3])

red_2 = Red([4, 5, 4, 3, 3])

red_3 = Red([4, 3, 3])

red_4 = Red([4, 3, 3, 3, 3])

red_5 = Red([4, 15, 5, 5, 3])

red_6  = Red([4, 6, 3])



#Entrenamos la red con el metodo "entrenar" que recibe:   El archivo de datos de entrenamiento;
#                                                         Cantidad maxima de epocas;
#                                                         Coeficiente de aprendizaje;
#                                                         Criterio de aceptacion;
#red.entrenar("TP2_grafica_comentarios/iris81_trn.csv", 1000, 0.01, 0.95)

#Ahora testeamos la red con el metodo "testear", enviandole el archivo con los datos de testeo
#red.testear("TP2_grafica_comentarios/iris81_tst.csv")

#Podemos graficar la evolucion del error por epoca con la siguiente funcion:
#red.evolucionError()

# Anda muy bien en general, a veces saca tasas de acierto de 95>% en datos de testing, y converge muy rápido. (Min épocas: 13)
#red_1.entrenar("TP2_grafica_comentarios/iris81_trn.csv", 1000, 0.01, 0.95)
#red_1.testear("TP2_grafica_comentarios/iris81_tst.csv")
#red_1.evolucionError()

# Converge un poco más lent0 que red_1 y la tasa de aciertos ronda el 90% en los datos de testing. (Min: 35 épocas)
#red_2.entrenar("TP2_grafica_comentarios/iris81_trn.csv", 1000, 0.01, 0.95)
#red_2.testear("TP2_grafica_comentarios/iris81_tst.csv")
#red_2.evolucionError()


# 1 sola capa oculta, anda generalmente lento.
red_3.entrenar("TP2_grafica_comentarios/iris81_trn.csv", 1000, 0.05, 0.95)
red_3.testear("TP2_grafica_comentarios/iris81_tst.csv")
red_3.evolucionError()

# Es el más lento de todos. A veces funciona un poquito bien.
# red_4.entrenar("TP2_grafica_comentarios/iris81_trn.csv", 1000, 0.03, 0.95)
# red_4.testear("TP2_grafica_comentarios/iris81_tst.csv")
# red_4.evolucionError()

# Bastante decente, tiene un número alto de tasas de aciertos y relativamente buena velocidad de convergencia.
#red_5.entrenar("TP2_grafica_comentarios/iris81_trn.csv", 1000, 0.01, 0.95)
#red_5.testear("TP2_grafica_comentarios/iris81_tst.csv")
#red_5.evolucionError()

# Bastante decente. Se dividen 6 las graficas presentadas con 6 planos.
#red_6.entrenar("TP2_grafica_comentarios/iris81_trn.csv", 1000, 0.01, 0.95)
#red_6.testear("TP2_grafica_comentarios/iris81_tst.csv")
#red_6.evolucionError()

# Observaciones generales:

# Velocidad de aprendizaje:
# Testeos en la red_5
# 0.03 Convergencia rápida pero menor tasa de aciertos (No pasa de 91)
# 0.01 anda bien, convergencia rápidas y tasa de aciertos muy alta.
# 0.001 Muy lento, se ajusta muy al final y toma muchas época (1000 épocas)
# 0.0001 No pudo converger en 1000 épocas.


# Testeos en la red_4
# 0.001 No converge.
# 0.03 Converge muy rápido.