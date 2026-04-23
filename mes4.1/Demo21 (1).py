import os, shutil
import pandas as pd
import cv2

print("Demo 21: Crear un DataSet de Sexo con el DataSet CelebA")

rutaOrigen = "C:/Data/Python/2026_01_IAG/Demos/datasets/img_align_celeba/"
rutaDestino = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Sexo/"
archivoAtributos = os.path.join(rutaOrigen, "list_attr_celeba.csv")
rutaImagenes = rutaOrigen + "img_align_celeba/img_align_celeba/"

print("1. Crear un DataFrame con los Atributos")
df = pd.read_csv(archivoAtributos)
print(df)

print("2. Leer los archivos con las imagenes")
archivos = os.listdir(rutaImagenes)
cM = 0
cF = 0
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
        sexo = "M" if dfFila["Male"].values[0]==1 else "F"
        print(f"Imagen: {archivo} - Sexo: {sexo}")    
        rutaDestinoMasculino = os.path.join(rutaDestino + "Masculino", archivo)
        rutaDestinoFemenino = os.path.join(rutaDestino + "Femenino", archivo)
        if(sexo=="M" and cM<nMuestras):
            cv2.imwrite(rutaDestinoMasculino, cara)
            cM+=1
        if(sexo=="F" and cF<nMuestras):
            cv2.imwrite(rutaDestinoFemenino, cara)
            cF+=1
        if(cM==nMuestras and cF==nMuestras):
            break
print(f"Archivos Masculino copiados: {cM}")
print(f"fArchivos Femenino copiados: {cF}")
