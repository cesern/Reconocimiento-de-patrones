"""
CESAR ERNESTO SALAZAR BUELNA
LIMPIEZA DE DATOS
"""
import os
import pandas as pd
import numpy as np

os.chdir('datos') 
#Lectura simple de datos
df = pd.read_csv("tiros.csv")


#IMPRIME LA INFORMACION DE CADA COLUMNA
print( df.info())

"""ANTES DE LA LIMPIEZA"""

#Da un resumen de los datos de cada columna como promedio, MAX, MIN, etc.
print( df.describe() )
#imprime TRUE o FALSE, dependiendo si hay nulos
print(df.isnull().any())
#COMO SHOT_CLOCK fue la que resulto con valores faltantes es la unica que se imprimio
#a detalle el procentaje
#de faltantes
SCPorcentajeNULL = df['SHOT_CLOCK'].isnull().sum() / df.shape[0] * 100
print ('Porcentaje de datos nulos en la columna SHOT_CLOCK',SCPorcentajeNULL,"%")

#pone NAN en los tiempos de toque cuando sean menores de 0
#df.loc[df["TOUCH_TIME"]<=0,"TOUCH_TIME"]=np.nan
df=df[df['TOUCH_TIME']>0]
df=df.dropna()

"""DESPUES DE LA LIMPIEZA"""

#Da un resumen de los datos de cada columna como promedio, MAX, MIN, etc.
print( df.describe() )
#imprime TRUE o FALSE, dependiendo si hay nulos
print(df.isnull().any())
#COMO SHOT_CLOCK fue la que resulto con valores faltantes es la unica que se imprimio
#a detalle el procentaje de faltantes
SCPorcentajeNULL = df['SHOT_CLOCK'].isnull().sum() / df.shape[0] * 100
print ('Porcentaje de datos nulos en la columna SHOT_CLOCK',SCPorcentajeNULL,"%")

#df=df.sample(10000)
#pasar los datos limpios a archivo para usar
df.to_csv("tirosL.csv", index=False)
