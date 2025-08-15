import matplotlib.pyplot as plt
import numpy as np

from PerceptronSimple import PerceptronSimple

# Se crea una clase llamada PerceptronSimpleGraficon que hereda de PerceptronSimple. Lo único que hace de distinto es graficar 
# cada 15 iteraciones en el loop de entrenamiento. 

# Se debe cerrar la pantalla del gráfico para que continúe.
class PerceptronSimpleGraficon(PerceptronSimple):
    counter = 0
    def update_w(self, x_datos, y_calculado, y_real):
        self.w = self.w + (self.gamma / 2)*float((int(y_real) - int(y_calculado))) * x_datos
        if (self.counter % 15 == 0):
            fig, ax = plt.subplots()             # Create a figure containing a single Axes.
            xc = np.linspace(-1, 1, 100)
            yes = []
            for k in xc:
                v = (self.w[0]/self.w[2]) - (self.w[1]/self.w[2])*k
                yes.append(v)
            ax.plot(xc, yes)  # Plot some data on the Axes.
            ax.plot(self.x_trn[:, 1], self.x_trn[:, 2], 'o')
            plt.show()  
        self.counter += 1
        
# Hacer False la siguiente variable si se quiere observar el comportamiento del XOR.
es_or = True
if (es_or):
    pOR = PerceptronSimpleGraficon("Guia1/OR_trn.csv", "Guia1/OR_trn.csv", "Guia1/OR_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)
    pOR.train()
else:
    pXOR = PerceptronSimple("Guia1/XOR_trn.csv", "Guia1/XOR_trn.csv", "Guia1/XOR_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)
    pXOR.train()





