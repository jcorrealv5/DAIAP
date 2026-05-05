import os, cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch

print("Demo 54: Crear DataSet con fotos y realizar Embedding a rostros, entrenar el Modelo para Agrupar y luego grabar")

print("1. Crear el DataSet de Caras de Alumnos")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=100, margin=0, min_face_size=20, device=device)
modelo = InceptionResnetV1(pretrained='vggface2').eval().to(device)
 
rutaAlumnos = "C:/Data/Python/2026_01_DAIAP/Imagenes/Alumnos/"
archivos = os.listdir(rutaAlumnos)
nArchivos = len(archivos)
listaX = []
imagenes = []
for i,archivo in enumerate(archivos):
    print(f"Archivo {i+1}/{nArchivos}: {archivo}")
    rutaArchivo = os.path.join(rutaAlumnos, archivo)
    imagen = cv2.imread(rutaArchivo, 1)
    caras = mtcnn(imagen)
    if caras is not None:     
        img_embeddings = modelo(caras.unsqueeze(0))
        imagenes.append(imagen)
        listaX.append(img_embeddings.squeeze(0).detach().numpy())
X = np.array(listaX)
print(f"Shape X: {X.shape}")

print("2. Crear el Modelo de Agrupamiento KMeans")
n_clusters = 3
modelo = KMeans(init="k-means++", n_clusters=n_clusters, random_state=0)

print("3. Entrenar el Modelo K-Medias y Agrupar")
y = modelo.fit_predict(X)
print("y: ", y.shape)

print("4. Guardar cada rostro en su respectivo grupo")
rutaSalida = "C:/Data/Python/2026_01_DAIAP/Imagenes/RostrosEmbedding/"
diccionario = {}
for i in range(X.shape[0]):
    grupo = "Grupo_" + str(y[i])
    imagen = imagenes[i]
    carpeta = rutaSalida + grupo
    if(not os.path.isdir(carpeta)):
        os.makedirs(carpeta)
        diccionario[grupo] = 0
    diccionario[grupo] += 1
    archivo = os.path.join(carpeta, str(diccionario[grupo]) + ".png")
    print("Guardando: ", archivo)
    cv2.imwrite(archivo, imagen)
print("Shape X: ", X.shape)
print("Shape X_pca: ", X.shape)
print("Estadistica: ", diccionario)