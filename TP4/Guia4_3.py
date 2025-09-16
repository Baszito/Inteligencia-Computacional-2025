import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import time

X = pd.read_csv('TP4/iris81_tst.csv', header=None).iloc[:,:-5].values

print("INERTIAS : ")
KMedias2=KMeans(2)
KMedias2.fit(X)
print(KMedias2.inertia_)

KMedias3=KMeans(3)
KMedias3.fit(X)
print(KMedias3.inertia_)

KMedias4=KMeans(4)
KMedias4.fit(X)
print(KMedias4.inertia_)

KMedias5=KMeans(5)
KMedias5.fit(X)
print(KMedias5.inertia_)

KMedias6=KMeans(6)
KMedias6.fit(X)
print(KMedias6.inertia_)

KMedias7=KMeans(7)
KMedias7.fit(X)
print(KMedias7.inertia_)

KMedias8=KMeans(8)
KMedias8.fit(X)
print(KMedias8.inertia_)

KMedias9=KMeans(9)
KMedias9.fit(X)
print(KMedias9.inertia_)

KMedias10=KMeans(10)
KMedias10.fit(X)
print(KMedias10.inertia_)

KMedias11=KMeans(11)
KMedias11.fit(X)
print(KMedias11.inertia_)

KMedias12=KMeans(12)
KMedias12.fit(X)
print(KMedias12.inertia_)

KMedias13=KMeans(13)
KMedias13.fit(X)
print(KMedias13.inertia_)

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
print(davies_bouldin_score(X,KMedias11.labels_))
print(davies_bouldin_score(X,KMedias12.labels_))
print(davies_bouldin_score(X,KMedias13.labels_))
