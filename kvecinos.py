# Inicializar el ambiente
import numpy as np
import pandas as pd
import math
import random
import time
import os
import sys
from scipy.spatial import distance
from sklearn import cluster
from matplotlib import pyplot as plt
np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1

os.chdir('datos')

# Leer los datos de archivo, separar training y test y calcular "prototipos de clase"
train_set = pd.read_csv("tirosL.csv")
train_set=train_set[list(['SHOT_NUMBER','PERIOD','SHOT_CLOCK','DRIBBLES','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST'])]
train_set=train_set.values
test_point = [3,4,12.1,14,11.9,14.6,1.8]
print("Datos de entrenaiento: \n{}\n\nDato de prueba:\n{}\n".format(train_set, test_point))

from sklearn import cluster
from matplotlib import pyplot as plt

num_clusters = 2
k_means = cluster.KMeans(n_clusters=num_clusters, init='random')
k_means.fit(train_set) 
print("Prototipos de clase (centroides):\n", k_means.cluster_centers_)
"""
fig = plt.figure(figsize=(8, 5))
colors = ['#ff0000', '#00ff00', '#0000ff']
for k in range(num_clusters):
    my_members = k_means.labels_ == k
    plt.plot(train_set[my_members, 2], train_set[my_members, 4], 'o', 
             markeredgecolor='k', markerfacecolor=colors[k], markersize=4)
    plt.plot(k_means.cluster_centers_[k][2], k_means.cluster_centers_[k][4], 'o', 
             markeredgecolor='k', markerfacecolor=colors[k], markersize=6)
plt.annotate('Punto nuevo', xy=(test_point[2], test_point[4]), xytext=(40, 50),
            arrowprops=dict(facecolor='black', shrink=0.1, width=1, headwidth=7))
plt.plot(test_point[2], test_point[4], 'w', marker='*', markersize=8)
plt.xlim([-10,110])
plt.ylim([-10,100])
plt.show()
"""
LARGER_DISTANCE = sys.maxsize

k_neighs = 172 # 5 vecinos... aunque tomaremos sólo el más cercano
neighbors_dists = [LARGER_DISTANCE] * k_neighs
neighbors = [0] * k_neighs
for i in range(len(train_set)):
    dist = distance.euclidean(train_set[i], test_point)
    for k in range(k_neighs):
        if (dist < neighbors_dists[k]) :
            for j in range(k_neighs-1, k, -1):
                neighbors_dists[j] = neighbors_dists[j-1]
                neighbors[j] = neighbors[j-1] 
            neighbors_dists[k] = dist
            neighbors[k] = i
            break
            
print("Los {} vecinos más próximos son:".format(k_neighs))
for k in range(k_neighs):
    clase = k_means.labels_[neighbors[k]]
    print("Vecino {}: {}, dist={}, clase={}, centroide={}"
          .format(k, neighbors[k], neighbors_dists[k], 
                  clase, k_means.cluster_centers_[clase]))
print("\nEl nuevo punto es asignado a la clase", k_means.labels_[neighbors[0]])


simple_vote = [0] * num_clusters
winner = 0 
for k in range(k_neighs):
    clase = k_means.labels_[neighbors[k]]
    simple_vote[clase] += 1
for k in range(num_clusters):
    if (simple_vote[k] == max(simple_vote)):
        winner = k
print("Votación simple:\nEl nuevo punto es asignado a la clase {} con {} vecinos cercanos.\n"
      .format(winner, simple_vote[winner]))

print("Los {} vecinos más próximos y sus pesos ponderados son:".format(k_neighs))
suma_dists = sum(neighbors_dists)
neighbors_weights = [0] * k_neighs
weighted_vote = [0] * num_clusters
winner = 0 
for k in range(k_neighs):
    neighbors_weights[k] = 1 - neighbors_dists[k] / suma_dists
    clase = k_means.labels_[neighbors[k]]
    weighted_vote[clase] += neighbors_weights[k]
    print("Vecino {}: peso={}, clase: {}"
          .format(k, neighbors_weights[k], k_means.labels_[neighbors[k]]))
for k in range(num_clusters):
    if (simple_vote[k] == max(simple_vote)):
        winner = k
print("\nVotación ponderada:")
print("El nuevo punto es asignado a la clase {} con una votación de {}."
      .format(winner, weighted_vote[winner]))

