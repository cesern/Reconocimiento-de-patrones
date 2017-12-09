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
print (df.describe())

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

K_INIT = 8
N_MIN = 1000
I_MAX = 10
S_MAX = 5
L_MIN = 80
P_MAX = 2

num_clusters = K_INIT # valor de k
iteration = 0

# Definir forma de muestreo; 0 = random, 1=head, 2=tail
SAMPLING_METHOD = 1 

centroids = []
def initialize_centroids():
    if (TALK) : 
        print("Centroides inicializados en:")
    for c in range(num_clusters):
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

elim = 0
members = []
def update_clusters():
    global num_clusters, elim, members
    changed = False
    
    if (TALK) :
        print("Actualizando clusters")
    for i in range(DATA_LEN):
        minDistance = LARGER_DISTANCE
        currentCluster = 0
        
        for j in range(num_clusters):
            dist = distance.euclidean(data[i].get_value(), centroids[j])
            if(dist < minDistance):
                minDistance = dist
                currentCluster = j
        
        if(data[i].get_cluster() is None or data[i].get_cluster() != currentCluster):
            data[i].set_cluster(currentCluster)
            changed = True
            
    members = [0] * num_clusters
    for i in range(DATA_LEN):
        members[data[i].get_cluster()] += 1
    
    if (TALK) : 
        for j in range(num_clusters):
            print("El cluster ", j, " incluye ", members[j], "miembros.")
        print()

    # Marcar los grupos a eliminar      
    to_eliminate = []
    for j in range(num_clusters):
        if (members[j] < N_MIN):
            to_eliminate.append(j)
                
    # Eliminar grupos marcados y recorrer índices
    elim = 0
    is_null = 0
    for j in to_eliminate: # to_eliminate contiene los índices originales a eliminar...
        if (TALK) :
            print("Eliminando cluster ", j)
        # Puesto que se han eliminado 'elim' clusters, los índices se han recorrido 
        # 'elim' veces; el verdero ínidice a eliminar es j_actual = j-elim
        j_actual = j - elim
        for i in range(DATA_LEN):
            cluster = data[i].get_cluster()
            if (cluster == j_actual) :
                data[i].set_cluster(None)
                is_null += 1
            elif (cluster != None and cluster > j_actual) :
                data[i].set_cluster(cluster-1)
        elim += 1
        for i in range(j_actual, num_clusters-1):
            members[i] = members[i + 1]
    
    num_clusters -= elim
    if (TALK and elim > 0) : 
        for j in range(num_clusters):
            print("El cluster ", j, " incluye ", members[j], "miembros.")
        print()

    return changed

# --------------------------
# Actualizar los clusters
KEEP_WALKING = update_clusters()
##############################
def update_centroids():
    global centroids
    centroids = []

    if (TALK) : 
        print("Los nuevos centroids son:")
    for j in range(num_clusters):
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
                means[i] = means[i] / clusterSize
            centroids.append(means)

        if (TALK) : 
            print(centroids[j])        
    if (TALK) : 
        print()
    
    return

# --------------------------
# Actualizar los centroides
update_centroids()
if (elim > 0) :
    KEEP_WALKING = update_clusters()
    update_centroids()
########
deltas = []
delta = 0
def update_deltas():
    global deltas, delta
    deltas = [0] * num_clusters
    delta = 0
    
    for i in range(DATA_LEN):
        cluster = data[i].get_cluster()
        deltas[cluster] += distance.euclidean(data[i].get_value(), centroids[cluster])
    mem = 0
    for i in range(num_clusters):
        delta += deltas[i]
        mem += members[i]
        deltas[i] /= members[i]
        if (TALK) : 
            print("Distancia promedio en el cluster {}:".format(i), deltas[i])        
    delta /= mem
    if (TALK) : 
        print("Distancia promedio global: {}\n".format(delta))
    
    return
    
update_deltas()

def divide_clusters():
    global num_clusters
    # Cálculo de desviaciones estandar
    sigma_vect = [[0] * DATA_SET.shape[1]] * num_clusters
    for d in range(DATA_LEN):
        cluster = data[d].get_cluster()
        p = data[d].get_value()
        for i in range(DATA_SET.shape[1]):
            sigma_vect[cluster][i] += (p[i] - centroids[cluster][i])**2        
    candidates = []
    for cluster in range(num_clusters):
        for i in range(DATA_SET.shape[1]):
            sigma_vect[cluster][i] = math.sqrt(sigma_vect[cluster][i]) / members[cluster]
            if (sigma_vect[cluster][i] > S_MAX):
                candidates.append(cluster)
                break # Sucio... pero eficiente :-) ... ya encontramos un atributo con elevada sigma
    
    divided = False
    for cluster in candidates:
        cond = num_clusters < K_INIT/2 or (deltas[cluster] > delta and members[cluster] > 2 * N_MIN)
        if(cond) :
            centroids.pop(cluster)
            points = []
            for d in range(DATA_LEN):
                if (data[d].get_cluster() == cluster):
                    points.append(data[d].get_value())
            dist = distance.squareform(distance.pdist(points, 'euclidean'))
            idx = (dist==dist.max()).argmax()
            z1 = list(points[idx // len(points)])
            z2 = list(points[idx % len(points)])
            if (TALK) :
                print("Se dividirá el cluster {}.\nSe crearán nuevos clusters en {} y {}.\n"
                     .format(cluster, z1, z2))
            centroids.append(z1)
            centroids.append(z2)
            num_clusters += 1
            divided = True
    
    if (divided) :
        if (TALK) : 
            print("Los nuevos centroids son:")
            for j in range(num_clusters):
                print(centroids[j])
            print("")

        update_clusters()
        update_centroids()
    
    return 

divide_clusters()

###
def mix_clusters():
    global centroids, num_clusters
    dist = distance.squareform(distance.pdist(centroids, 'euclidean'))
    flag = math.floor(dist.max() * 10)
    dist[dist == 0] = flag
    
    mixed = False
    while (dist.min() < flag):
        idx = (dist==dist.min()).argmax()
        z1 = idx // len(centroids)
        z2 = idx % len(centroids)
        
        if (dist.min() < L_MIN):
            dist[z1] = flag
            dist[:,z1] = flag
            dist[z2] = flag
            dist[:,z2] = flag
            z = [sum(x)/2 for x in zip(centroids[z1], centroids[z2])]
            centroids[z1] = z
            centroids[z2] = [LARGER_DISTANCE]*DATA_SET.shape[1]
            num_clusters -= 1

            mixed = True
            if(TALK):
                print("Unificando clusters {} y {}\nSe creará nuevo centroide en {}\n"
                      .format(z1, z2, z))
        else :
            dist[z1][z2] = flag
            dist[z2][z1] = flag
        
    if (mixed) :
        update_clusters()
        update_centroids()

    return

mix_clusters()

# Reproducido aquí para facilitar la ejecución
iteration +=1
while(iteration < I_MAX and KEEP_WALKING) :
    if (num_clusters <= K_INIT / 2) :
        divide_clusters()
    elif (iteration % 2 == 0 or num_clusters > 2 * K_INIT) :
        mix_clusters()
        
    KEEP_WALKING = update_clusters()
    if (KEEP_WALKING):
        update_centroids()
    else :
        if (TALK) : 
            print ("No más cambios.")
    iteration += 1

