import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import time

X = pd.read_csv('TP4/iris81_tst.csv', header=None).iloc[:,:-5].values

inertias = []

print("INERTIAS : ")
KMedias2=KMeans(2)
KMedias2.fit(X)
inertias.append([2,KMedias2.inertia_])
print(KMedias2.inertia_)

KMedias3=KMeans(3)
KMedias3.fit(X)
inertias.append([3,KMedias3.inertia_])
print(KMedias3.inertia_)

KMedias4=KMeans(4)
KMedias4.fit(X)
inertias.append([4,KMedias4.inertia_])
print(KMedias4.inertia_)

KMedias5=KMeans(5)
KMedias5.fit(X)
inertias.append([5,KMedias5.inertia_])
print(KMedias5.inertia_)

KMedias6=KMeans(6)
KMedias6.fit(X)
inertias.append([6,KMedias6.inertia_])
print(KMedias6.inertia_)

KMedias7=KMeans(7)
KMedias7.fit(X)
inertias.append([7 ,KMedias7.inertia_])
print(KMedias7.inertia_)

KMedias8=KMeans(8)
KMedias8.fit(X)
inertias.append([8,KMedias8.inertia_])
print(KMedias8.inertia_)

KMedias9=KMeans(9)
KMedias9.fit(X)
inertias.append([9,KMedias9.inertia_])
print(KMedias9.inertia_)

KMedias10=KMeans(10)
KMedias10.fit(X)
inertias.append([10,KMedias10.inertia_])
print(KMedias10.inertia_)


print("DAVIS BOULDIN: ")
print(davies_bouldin_score(X,KMedias2.labels_))
print(davies_bouldin_score(X,KMedias3.labels_))
print(davies_bouldin_score(X,KMedias4.labels_))
print(davies_bouldin_score(X,KMedias5.labels_))
print(davies_bouldin_score(X,KMedias6.labels_))
print(davies_bouldin_score(X,KMedias7.labels_))
print(davies_bouldin_score(X,KMedias8.labels_))
print(davies_bouldin_score(X,KMedias9.labels_))
print(davies_bouldin_score(X,KMedias10.labels_))

fig, ax = plt.subplots()

# Recorremos pares de puntos
for i in range(len(inertias)-1):
    # Línea entre puntos
    ax.plot([inertias[i][0], inertias[i+1][0]],
            [inertias[i][1], inertias[i+1][1]],
            '-', color='blue')  # línea azul
    
    # Marcador 'x' en cada punto
    ax.plot(inertias[i][0], inertias[i][1],
            'x', markeredgewidth=2, color='red')

# También ploteo el último punto
ax.plot(inertias[-1][0], inertias[-1][1], 'x', markeredgewidth=2, color='red')

# Límites y ticks
ax.set(xlim=(0, 12), xticks=np.arange(1, 12),
       ylim=(0, 12), yticks=np.arange(1, 12))

# Agregar grilla
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title("Compactitud con distintos K")
plt.show()

