import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

os.chdir('datos')
datos = pd.read_csv("tirosL.csv")

#datos=shuffle(datos.iloc[0:100000])
#clases
Y=datos.iloc[:,7].values
#caracteristicas
X=datos.iloc[:,0:7].values
#aprendizaje
eta=0.01
#pesos iniciales
w=np.zeros(X.shape[1])#20 ceros
#datos para entrenamiento y prueba
train_data, test_data, train_y, test_y=train_test_split(X,Y,test_size=0.5)

#entrenamiento
vector=1
for xi, target in zip(train_data, train_y):
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    error = target - output
    w += eta * error * xi
    #if(error!=0):
        #print(vector,"\t",target,"\t",output,"\t",error,"\t",w,"\n")
    vector+=1

#print("Pesos Finales: ",w)
# Prueba
errores = 0
for xi, target in zip(test_data, test_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_y), 
                                                        errores/len(test_y)*100))
print("SEGUNDA RONDA con la mitad de los datos")
#FALTA CORREGIR DE AQUI EN ADELantE LAS variabLES NOmbres
# 2a ronda de entrenamiento
vectores = 0
shuffled_data = shuffle(list(zip(train_data, train_y)))
for xi, target in shuffled_data:
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    error = target - output
    w += eta * error * xi
    if (target != output) :
        vectores += 1
print("{} vectores de entrenamiento mal clasificados de {} ({}%)".
      format(vectores, len(train_y), vectores/len(train_y)*100))
    
# Prueba
errores = 0
for xi, target in zip(test_data, test_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_y), 
                                                        errores/len(test_y)*100))
print("TERCERA RONDA con datos completos")
# 3a ronda de entrenamiento... con todos los datos
vectores = 0
for xi, target in zip(X, Y):
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    error = target - output
    w += eta * error * xi
    if (target != output) :
        vectores += 1
print("{} vectores de entrenamiento mal clasificados de {} ({}%)".
      format(vectores, len(Y), vectores/len(Y)*100))
    
# Prueba
errores = 0
for xi, target in zip(test_data, test_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_y), 
                                                        errores/len(test_y)*100))