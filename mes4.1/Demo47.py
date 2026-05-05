import os, cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

print("Demo 47: Usar el Algoritmo PCA para Visualizar en 2D el DataSet de Sexo")

print("1. Crear el DataSet para el Sexo con las Caras de Hombres y Mujeres")
rutaSexo = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Sexo/"
listaX = []
listaY = []
carpetas = os.listdir(rutaSexo)
for carpeta in carpetas:
    sexo = carpeta[0]
    print("_" * 100)
    print(f"Categoria: {carpeta}")
    print("_" * 100)
    archivos = os.listdir(rutaSexo + carpeta)
    for archivo in archivos:
        print(archivo)
        rutaArchivo = os.path.join(rutaSexo + carpeta, archivo)
        imagenGris = cv2.imread(rutaArchivo, 0)
        imagenGris100 = cv2.resize(imagenGris, (100,100))
        imagenPlana = imagenGris100.flatten()
        listaX.append(imagenPlana)
        listaY.append(sexo)
X = np.array(listaX)
y = np.array(listaY)
print("Shape X: ", X.shape)
print("Shape y: ", y.shape)

print("2. Reducir de 10,000 Dimensiones el Digito a solo 2")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Shape X_pca: ", X_pca.shape)
print("Primera Foto X_pca: ", X_pca[0])

print("3. Graficar la data reducida")
serieX_Femenino = X_pca[y=="F",0]
nMujeres = serieX_Femenino.shape[0]
serieX_Masculino = X_pca[y=="M",0]
nHombres = serieX_Masculino.shape[0]
plt.scatter(serieX_Femenino,X_pca[y=="F",1],color="red",label="Mujeres: " + str(nMujeres))
plt.scatter(serieX_Masculino,X_pca[y=="M",1],color="blue",label="Hombres: " + str(nHombres))
plt.legend()
plt.show()