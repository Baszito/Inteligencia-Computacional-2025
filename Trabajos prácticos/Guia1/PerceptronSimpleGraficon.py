# Se crea una clase llamada PerceptronSimpleGraficon que hereda de PerceptronSimple. Lo único que hace de distinto es graficar 
# cada 15 iteraciones en el loop de entrenamiento. 
from PerceptronSimple import PerceptronSimple
import matplotlib.pyplot as plt
import pandas as pandas
import numpy as np

# Se debe cerrar la pantalla del gráfico para que continúe.
class PerceptronSimpleGraficon(PerceptronSimple):
    counter = 0
    def update_w(self, x_datos, y_calculado, y_real):
        self.w = self.w + (self.gamma / 2)*float((int(y_real) - int(y_calculado))) * x_datos
        if (self.counter % 50 == 0):
            fig, ax = plt.subplots()             # Create a figure containing a single Axes.
            xc = np.linspace(-1, 1, 100)
            yes = []
            for k in xc:
                v = (self.w[0]/self.w[2]) - (self.w[1]/self.w[2])*k
                yes.append(v)
            print(self.w)
            ax.grid(True)
            ax.plot(xc, yes)  # Plot some data on the Axes.
            ax.plot(self.x_trn[:, 1], self.x_trn[:, 2], 'o')
            plt.show()  
        self.counter += 1