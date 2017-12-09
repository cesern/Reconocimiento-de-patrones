"""
CESAR ERNESTO SALAZAR BUELNA
LIMPIEZA DE DATOS
"""
import os
import pandas as pd
from scipy.spatial import distance
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
np.set_printoptions(precision=1, suppress=True) # Cortar la impresión de decimales a 1

os.chdir('datos') 
#Lectura simple de datos ya con la limpieza
df = pd.read_csv("tirosL.csv")
#los datos ya estan limpios

#df=df.sample(10000)

X = df.head(1000)

# Convertir el vector de distancias a una matriz cuadrada
md = distance.squareform(distance.pdist(X, 'euclidean')) 
print(md)

Z = linkage(X, 'complete')
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=5, show_leaf_counts=True, leaf_font_size=14.)
#dendrogram(Z, leaf_font_size=14)
plt.show()