import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
train_data, test_data, train_y, test_y=train_test_split(X,Y,test_size=0.2)

#X_trainPIDff, X_testPIDff, y_trainPIDff, y_testPIDff = train_test_split(
    #df_pureAd, df_classAd, test_size=.2)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='tanh', #identity,logistic,tanh,relu
                    hidden_layer_sizes=(2,2,1), random_state=1, 
                    learning_rate_init=0.001, max_iter=5000)

clf.fit(train_data,train_y)                         

# Prueba
errores = 0
for xi, target in zip(test_data, test_y) :
    output = clf.predict(xi.reshape(1, -1))
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_data), 
                                                        errores/len(test_data)*100))