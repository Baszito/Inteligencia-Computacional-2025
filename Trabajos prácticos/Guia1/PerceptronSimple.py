from Int import Int
from sgn import sgn

# Guía rápida de como usar la clase Int

# 1) Se define una clase y se hace que herede de "Int"
class PerceptronSimple(Int):
    # 2)
    # Se implementan las funciones "sigmoid" y "update_w"
    def sigmoid(self, v):
        return sgn(v)
    def update_w(self, x_datos, y_calculado, y_real):
        self.w = self.w + (self.gamma / 2)*float((int(y_real) - int(y_calculado))) * x_datos
        
# 3) Se crea una instancia con las rutas de Training, Validación y Testing respectivamente. En este caso usamos los datos de Training en validación.
# Opcionalmente se definen los parámetros gamma, max_iterations y success_rate
perceptron = PerceptronSimple("Guia1/OR_trn.csv", "Guia1/OR_trn.csv", "Guia1/OR_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)

# 4) Se llama al método train() para entrenar el perceptrón.
#perceptron.train()

# 5) Se llama al método test_data() para probar sobre los datos de testing.
#perceptron.test_data()