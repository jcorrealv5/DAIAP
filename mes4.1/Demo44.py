from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

print("Demo 44: Usar el Algoritmo PCA para Visualizar en 2D los Digitos de MNIST")

print("1. Cargar el Dataset de MNIST-64")
dst = load_digits()
X = dst["data"]
y = dst["target"]
print("Shape X: ", X.shape)
print("Primer Digito X: ", X[0])

print("2. Reducir de 64 Dimensiones el Digito a solo 2")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Shape X_pca: ", X_pca.shape)
print("Primer Digito X_pca: ", X_pca[0])

print("3. Graficar la data reducida")
for i in range(10):
    my_members = y == i
    plt.scatter(X_pca[my_members,0],X_pca[my_members,1],label=i)
plt.legend()
plt.show()