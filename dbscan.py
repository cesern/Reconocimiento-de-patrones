# Inicializar el ambiente
import numpy as np
import pandas as pd
import os
from sklearn import cluster
from sklearn import metrics
from matplotlib import pyplot as plt

def muestra(multishapes):
    plt.plot(multishapes[:, 0], multishapes[:, 1], 'o', 
            markeredgecolor='0', markerfacecolor='0', markersize=5)
    plt.show()

def k_means(multishapes, num_clusters):
    k_means = cluster.KMeans(n_clusters=num_clusters, init='random')
    k_means.fit(multishapes)
    fig = plt.figure(figsize=(8, 6))
    colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ffff00', '#f6ff00', 
            '#2f800f', '#a221b5', '#21b5ac', '#b1216c','#D51010', '#141BDF', '#15D112', '#BDD10F', '#F03C92', '#74ECEE',]
    for k in range(num_clusters):
        my_members = k_means.labels_ == k
        plt.plot(multishapes[my_members, 0], multishapes[my_members, 1], 'o', 
                markeredgecolor='k', markerfacecolor=colors[k], markersize=5)
    plt.show()            

def dbscan(multishapes):
    db = cluster.DBSCAN(eps=0.3, min_samples=20)
    db.fit(multishapes)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    unique_labels = set(db.labels_)
    print(unique_labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(unique_labels) - (1 if -1 in db.labels_ else 0)

    fig = plt.figure(figsize=(8, 6))
    colors = ['#D51010', '#141BDF', '#15D112', '#BDD10F', '#F03C92', '#74ECEE', '#f6ff00', 
            '#2f800f', '#a221b5', '#21b5ac', '#b1216c']
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        my_members = db.labels_ == k

        xy = multishapes[my_members & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', 
                markerfacecolor=col, markeredgecolor='k', markersize=11)

        xy = multishapes[my_members & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', 
                markerfacecolor=col, markeredgecolor='k', markersize=6)
        
    plt.title('Número estimado de clusters: %d' % n_clusters_)
    plt.show()

if __name__ == '__main__':
    os.chdir('datos')
    multishapes = pd.read_csv("tirosL.csv")
    multishapes=multishapes[['SHOT_CLOCK', 'CLOSE_DEF_DIST']].values
    #multishapes=multishapes.values
    #multishapes = pd.read_csv("tirosMuestra.csv").values
    #muestra(multishapes)
    k_means(multishapes, 12)
    dbscan(multishapes)