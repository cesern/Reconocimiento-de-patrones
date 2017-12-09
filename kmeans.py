# Inicializar el ambiente
import numpy as np
import pandas as pd
import math
import random
import time
import os
import sys
from scipy.spatial import distance
np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1

os.chdir('datos')
LARGER_DISTANCE = sys.maxsize
TALK = True # TALK = True, imprime resultados parciales

# Leer los datos de archivo
df = pd.read_csv("tirosL.csv")
#pone NAN en los tiempos de toque cuando sean menores de 0
df.loc[df["TOUCH_TIME"]<=0,"TOUCH_TIME"]=np.nan
#quita los faltantes
df=df.dropna()
DATA_SET = df.values
DATA_LEN = len(DATA_SET)

# Definir una clase para expresar puntos y su asignación a un cluster
class DataPoint:
    def __init__(self, p):
        self.value = p[:]
        
    def set_value(self, p):
        self.value = p
    
    def get_value(self):
        return self.value
    
    def set_cluster(self, cluster):
        self.cluster = cluster
    
    def get_cluster(self):
        return self.cluster

data = []
def initialize_dataset():
    for i in range(DATA_LEN):
        point = DataPoint(DATA_SET[i])
        point.set_cluster(None)
        data.append(point)
    return


# --------------------------
# Crear el conjunto de datos
initialize_dataset()

NUM_CLUSTERS = 5

# Definir forma de muestreo; 0 = random, 1=head, 2=tail
SAMPLING_METHOD = 0

centroids = []
def initialize_centroids():
    if (TALK) : 
        print("Centroides inicializados en:")
    for c in range(NUM_CLUSTERS):
        if (SAMPLING_METHOD == 0) :
            which = random.randint(0,DATA_LEN-1)
        elif (SAMPLING_METHOD == 1):
            which = c
        else :
            which = DATA_LEN-1 - c
                
        centroids.append(list(DATA_SET[which]))
        if (TALK) : 
            print(centroids[c])        
    if (TALK) : 
        print()
    
    return

# --------------------------
# Inicializar los centroides
initialize_centroids()

def update_clusters():
    changed = False
    
    for i in range(DATA_LEN):
        minDistance = LARGER_DISTANCE
        currentCluster = 0
        
        for j in range(NUM_CLUSTERS):
            dist = distance.euclidean(data[i].get_value(), centroids[j])
            if(dist < minDistance):
                minDistance = dist
                currentCluster = j
        
        if(data[i].get_cluster() is None or data[i].get_cluster() != currentCluster):
            data[i].set_cluster(currentCluster)
            changed = True
            
    members = [0] * NUM_CLUSTERS
    for i in range(DATA_LEN):
        members[data[i].get_cluster()] += 1
    
    if (TALK) : 
        for j in range(NUM_CLUSTERS):
            print("El cluster ", j, " incluye ", members[j], "miembros.")
        print()
            
    return changed

# --------------------------
# Actualizar los clusters
KEEP_WALKING = update_clusters()

def update_centroids():    
    if (TALK) : 
        print("Los nuevos centroids son:")
    for j in range(NUM_CLUSTERS):
        means = [0] * DATA_SET.shape[1]
            
        clusterSize = 0
        for k in range(len(data)):
            if(data[k].get_cluster() == j):
                p = data[k].get_value()
                for i in range(DATA_SET.shape[1]):
                    means[i] += p[i]
                clusterSize += 1

        if(clusterSize > 0):
            for i in range(DATA_SET.shape[1]):
                centroids[j][i] = means[i] / clusterSize

        if (TALK) : 
            print(centroids[j])        
    if (TALK) : 
        print()
    
    return

# --------------------------
# Actualizar los centroides
update_centroids()


while(KEEP_WALKING):
    KEEP_WALKING = update_clusters()
    if (KEEP_WALKING):
        update_centroids()
    else :
        if (TALK) : 
            print ("No más cambios.")