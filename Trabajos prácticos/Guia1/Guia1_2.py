import matplotlib.pyplot as plt
import numpy as np

from PerceptronSimpleGraficon import PerceptronSimpleGraficon


        
# Hacer False la siguiente variable si se quiere observar el comportamiento del XOR.
es_or = False
if (es_or):
    pOR = PerceptronSimpleGraficon("Guia1/OR_trn.csv", "Guia1/OR_trn.csv", "Guia1/OR_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)
    pOR.train()
else:
    pXOR = PerceptronSimpleGraficon("Guia1/XOR_trn.csv", "Guia1/XOR_trn.csv", "Guia1/XOR_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)
    pXOR.train()





