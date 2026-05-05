from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

print("Demo 46: Usar el Algoritmo PCA para Visualizar en 2D el DataSet de Diabetes")

print("1. Cargar el Dataset de Diabetes")
dst = load_diabetes()
X = dst["data"]
y = dst["target"]
print("Shape X: ", X.shape)
print("Primera Muestra X: ", X[0])

print("2. Reducir de 10 Dimensiones a solo 2")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Shape X_pca: ", X_pca.shape)
print("Primera Muestra X_pca: ", X_pca[0])

print("3. Graficar la data reducida")
umbralDiabetes = 125
serieX_No = X_pca[y<umbralDiabetes,0]
serieY_No = X_pca[y<umbralDiabetes,1]
serieX_Si = X_pca[y>=umbralDiabetes,0]
serieY_Si = X_pca[y>=umbralDiabetes,1]
cantidadNo = serieX_No.shape[0]
cantidadSi = serieX_Si.shape[0]
plt.scatter(serieX_No,serieY_No,color="green",label="No Tienen: " + str(cantidadNo))
plt.scatter(serieX_Si,serieY_Si,color="red",label="Si Tienen: " + str(cantidadSi))
plt.legend()
plt.show()