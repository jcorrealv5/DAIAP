import os, cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("Demo 50: Crear un Modelo para Agrupar Rostros usando K-Medias pero antes usar PCA para reducir dimensiones a 500")
archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)

def crearData():
    listaX = []
    listaY = []
    carpetaRaiz = "C:/Data/Python/2026_01_DAIAP/DataSets/UTKFace/test/"
    archivos = os.listdir(carpetaRaiz)
    for archivo in archivos:
        campos = archivo.split("_")      
        if(len(campos)==4):
            edad = int(campos[0])
            if(edad>=18 and edad<=70):          
                print(archivo)
                rutaArchivo = os.path.join(carpetaRaiz, archivo)
                imagenGris = cv2.imread(rutaArchivo, 0)
                caras = clasificador.detectMultiScale(imagenGris, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                for(x,y,w,h) in caras:
                    cara = imagenGris[y:y+h, x:x+w]
                    nCaras = len(caras)
                    if(nCaras>0):
                        imgResize = cv2.resize(imagenGris, (100, 100))
                        imagenPlana = imgResize.flatten()
                        listaX.append(imagenPlana)
        else:
            break
    X = np.array(listaX)
    return X

print("1. Crear el DataSet para Agrupar Rostros")
X = crearData()

print("2. Reducir la Dimensionalidad con PCA")
pca = PCA(n_components=500, whiten=True).fit(X)
X_pca = pca.transform(X)

print("3. Crear el Modelo de Agrupamiento KMeans")
n_clusters = 10
modelo = KMeans(init="k-means++", n_clusters=n_clusters, random_state=0)

print("4. Entrenar el Modelo K-Medias y Agrupar")
y = modelo.fit_predict(X_pca)
print("y: ", y.shape)

print("4. Guardar cada rostro en su respectivo grupo")
rutaSalida = "C:/Data/Python/2026_01_DAIAP/Imagenes/RostrosPCA/"
diccionario = {}
for i in range(X.shape[0]):
    grupo = "Grupo_" + str(y[i])
    imagen = X[i].reshape(100,100)
    carpeta = rutaSalida + grupo
    if(not os.path.isdir(carpeta)):
        os.makedirs(carpeta)
        diccionario[grupo] = 0
    diccionario[grupo] += 1
    archivo = os.path.join(carpeta, str(diccionario[grupo]) + ".png")
    print("Guardando: ", archivo)
    cv2.imwrite(archivo, imagen)
print("Shape X: ", X.shape)
print("Shape X_pca: ", X_pca.shape)
print("Estadistica: ", diccionario)