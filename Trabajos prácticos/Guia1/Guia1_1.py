from PerceptronSimple import PerceptronSimple

# Se crea una instancia de la clase Perceptron Simple, que utiliza la función sgn(x) como sigmoide.
perceptron = PerceptronSimple("Guia1/OR_trn.csv", "Guia1/OR_trn.csv", "Guia1/OR_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)

# Se invoca el método train() que entrena los datos datos en los archivos .csv
perceptron.train()

print(perceptron.w)

# Una vez entrenado el perceptrón, se prueba con los datos de testing.
perceptron.test_data()