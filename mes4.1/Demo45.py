from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

print("Demo 45: Usar el Algoritmo PCA para Visualizar en 2D el DataSet Iris")

print("1. Cargar el Dataset de Iris")
dst = load_iris()
X = dst["data"]
y = dst["target"]
etiquetas = dst["target_names"]
print("Shape X: ", X.shape)
print("Primer Iris X: ", X[0])
print("Etiquetas: ", etiquetas)

print("2. Reducir de 4 Dimensiones el Iris a solo 2")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Shape X_pca: ", X_pca.shape)
print("Primer Iris X_pca: ", X_pca[0])

print("3. Graficar la data reducida")
for i in range(3):
    my_members = y == i
    plt.scatter(X_pca[my_members,0],X_pca[my_members,1],label=etiquetas[i])
plt.legend()
plt.show()