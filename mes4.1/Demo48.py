import os, cv2
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

print("Demo 48: Crear un Modelo para Agrupar Digitos de MNIST-784 usando K-Medias y guardar a disco")

print("1. Cargar el DataSet de Digitos MNIST-784")
dst = fetch_openml("mnist_784", as_frame=False)
X = dst["data"]
print("Shape X: ", X.shape)

print("2. Crear el Modelo de Agrupamiento KMeans")
n_clusters = 10
modelo = KMeans(n_clusters=n_clusters, random_state=0)

print("3. Entrenar el Modelo K-Medias y Agrupar")
y = modelo.fit_predict(X)
print("y: ", y.shape)

print("4. Guardar cada digito en su respectivo grupo")
rutaSalida = "C:/Data/Python/2026_01_DAIAP/Imagenes/Digitos/"
diccionario = {}
for i in range(X.shape[0]):
    grupo = "Grupo_" + str(y[i])
    imagen = X[i].reshape(28,28)
    carpeta = rutaSalida + grupo
    if(not os.path.isdir(carpeta)):
        os.makedirs(carpeta)
        diccionario[grupo] = 0
    diccionario[grupo] += 1
    archivo = os.path.join(carpeta, str(diccionario[grupo]) + ".png")
    print("Guardando: ", archivo)
    cv2.imwrite(archivo, imagen)
print("Shape X: ", X.shape)
print("Estadistica: ", diccionario)