from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

print("Demo 43: Crear un Modelo para Agrupar Digitos de MNIST usando K-Medias")

print("1. Cargar el DataSet de Digitos MNIST-64")
dst = load_digits()
print("Keys: ", dst.keys())
X = dst["data"]
imagenes = dst["images"]
print("Shape X: ", X.shape)

print("2. Crear el Modelo de Agrupamiento KMeans")
n_clusters = 10
modelo = KMeans(n_clusters=n_clusters, random_state=0)

print("3. Entrenar el Modelo SVC")
y = modelo.fit_predict(X)
print("y: ", y.shape)
valores = np.bincount(y)
print("Valores: ", valores)

print("4. Mostrar y Graficar el Primer Digito")
print("Grupo del #0: ", y[0])
plt.imshow(imagenes[0], cmap="gray")
plt.show()

print("5. Graficando los 10 Grupos")
centros = modelo.cluster_centers_.reshape(n_clusters,8,8)
print("centros: ", centros.shape)
fig, ejes = plt.subplots(2, 5)
for eje, centro in zip(ejes.flat, centros):
    eje.imshow(centro, cmap="gray")
plt.show()