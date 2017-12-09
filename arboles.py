import numpy as np
import pandas as pd
from IPython.display import Image, display  
from sklearn import tree
# Para hacer un muestreo aleatorio
from sklearn.model_selection import train_test_split

classes_names = ['NO ENTRO','ENTRO']
feats_names = ['SHOT_NUMBER','PERIOD','SHOT_CLOCK','DRIBBLES','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST']
db = pd.read_csv("datos/tirosMuestra1000.csv")
display(db.describe())

etiqueta=db['SHOT_RESULT']
db=db.drop('SHOT_RESULT',1)

train_features, test_features, train_targets, test_targets = train_test_split(
    db.values, etiqueta.values.ravel(), test_size=0.1)
#print("{}\n{}\n{}".format(classesDf.head(5), #Dataframe
#                          classesDf.head(5).values, #NumpyArray (arreglo de arreglos)
#                          classesDf.head(5).values.ravel()) #NumpyArray (arreglo)
#     )

test_targets = list(test_targets)
train_targets = list(train_targets)

#print ("Clases de la muestra de prueba: ", test_targets)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features, train_targets)

predictions_test = clf.predict(test_features)
fails_test = np.sum(test_targets != predictions_test)
print("Objetivos: ", test_targets)
print("Resultados: ", list(predictions_test))
print("Puntos mal clasificados en el conjunto de prueba: {} de {} ({}%)\n"
      .format(fails_test, len(test_targets), 100*fails_test/len(test_targets)))

import pydotplus # brew install graphviz, pip install pydotplus
from io import StringIO
from IPython.display import Image, display  

dotfile = StringIO()
tree.export_graphviz(clf, out_file=dotfile, class_names=classes_names, 
                     feature_names=feats_names,
                     filled=False, rounded=True)
graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png('ArbolTotal.png')
display(Image(graph.create_png()))