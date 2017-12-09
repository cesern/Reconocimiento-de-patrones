import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
import os
import time
from IPython.display import display

os.chdir('datos')
datos = pd.read_csv("tirosMuestra10000.csv")

etiquetas=datos['SHOT_RESULT']
datos=datos.drop('SHOT_RESULT',1)

x_train, x_test, y_train, y_test=train_test_split(datos,etiquetas,test_size=0.2)
#LINEAL VARIANDO C
for i in [.1,.5,1,10,100]:
    svmLineal=LinearSVC(C=i)
    start_time=time.time()
    svmLineal.fit(x_train,y_train)
    elapsed_time=time.time() - start_time

    #make an array...
    preds_Lineal=svmLineal.predict(x_test)
    fails_Lineal=np.sum(y_test != preds_Lineal)
    preds_train_Lineal=svmLineal.predict(x_train)
    fails_train_Lineal=np.sum(y_train!=preds_train_Lineal)
    print("SVM Lineal, C={}\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
        \nPuntos mal clasificados (prueba): {} de {} ({}%)\
        \nAciertos del {}%\nTiempo: {}\n"
        .format(i,fails_train_Lineal, len(y_train), 100*fails_train_Lineal/len(y_train),
                fails_Lineal, len(y_test), 100*fails_Lineal/len(y_test), 
                svmLineal.score(x_test, y_test)*100, elapsed_time))
#RBF VARIANDO C
for i in [.1,.5,1,10,100]:
    svmRbf = SVC(kernel='rbf', C=i)
    start_time = time.time()
    svmRbf.fit(x_train, y_train)
    elapsed_time = time.time() - start_time

    preds_train_Rbf = svmRbf.predict(x_train)
    fails_train_Rbf = np.sum(y_train != preds_train_Rbf)

    preds_Rbf = svmRbf.predict(x_test)
    fails_Rbf = np.sum(y_test != preds_Rbf)

    print("SVM RBF, C={}\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
        \nPuntos mal clasificados (prueba): {} de {} ({}%)\
        \nAciertos del {}%\nTiempo: {}\n"
        .format(i,fails_train_Rbf, len(y_train), 100*fails_train_Rbf/len(y_train),
                fails_Rbf, len(y_test), 100*fails_Rbf/len(y_test), 
                svmRbf.score(x_test, y_test)*100, elapsed_time)) 
#RBF VARIANDO GAMMA
for i in [.1,.3,.5,.7,1]:
    svmRbf = SVC(kernel='rbf', C=.5,gamma=i)
    start_time = time.time()
    svmRbf.fit(x_train, y_train)
    elapsed_time = time.time() - start_time

    preds_train_Rbf = svmRbf.predict(x_train)
    fails_train_Rbf = np.sum(y_train != preds_train_Rbf)

    preds_Rbf = svmRbf.predict(x_test)
    fails_Rbf = np.sum(y_test != preds_Rbf)

    print("SVM RBF, C={}\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
        \nPuntos mal clasificados (prueba): {} de {} ({}%)\
        \nAciertos del {}%\nTiempo: {}\n"
        .format(i,fails_train_Rbf, len(y_train), 100*fails_train_Rbf/len(y_train),
                fails_Rbf, len(y_test), 100*fails_Rbf/len(y_test), 
                svmRbf.score(x_test, y_test)*100, elapsed_time)) 
                                            