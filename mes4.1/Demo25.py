import os, shutil
import pandas as pd
import cv2

print("Demo 25: Crear un DataSet de Lentes con el DataSet CelebA")

rutaOrigen = "C:/Data/Python/2026_01_IAG/Demos/datasets/img_align_celeba/"
rutaDestino = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Lentes/"
archivoAtributos = os.path.join(rutaOrigen, "list_attr_celeba.csv")
rutaImagenes = rutaOrigen + "img_align_celeba/img_align_celeba/"

print("1. Crear un DataFrame con los Atributos")
df = pd.read_csv(archivoAtributos)
print(df)

print("2. Leer los archivos con las imagenes")
archivos = os.listdir(rutaImagenes)
cC = 0
cS = 0
nMuestras = 1000
archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)
for archivo in archivos:
    rutaOrigenImagen = os.path.join(rutaImagenes, archivo)
    imagen = cv2.imread(rutaOrigenImagen)
    caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    nCaras = len(caras)
    if(nCaras>0):
        for(x,y,w,h) in caras:
            cara = imagen[y:y+h, x:x+w]
        dfFila = df[df["image_id"]==archivo]
        lentes = "C" if dfFila["Eyeglasses"].values[0]==1 else "S"
        print(f"Imagen: {archivo} - Lentes: {lentes}") 
        rutaDestinoConLentes = os.path.join(rutaDestino + "Con Lentes", archivo)
        rutaDestinoSinLentes = os.path.join(rutaDestino + "Sin Lentes", archivo)
        if(lentes=="C" and cC<nMuestras):
            cv2.imwrite(rutaDestinoConLentes, cara)
            cC+=1
        if(lentes=="S" and cS<nMuestras):
            cv2.imwrite(rutaDestinoSinLentes, cara)
            cS+=1
        if(cC==nMuestras and cS==nMuestras):
            break
print(f"Archivos Con Lentes copiados: {cC}")
print(f"fArchivos Sin Lentes copiados: {cS}")
