import pandas as pd
import cv2, os

rutaImagenes = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/img_align_celeba/"
archivoAtributos = r"C:\Data\Python\2026_01_DAIAP\DataSets\CelebA\list_attr_celeba.csv"
rutaFemenino = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Sexo/Femenino/"
rutaMasculino = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Sexo/Masculino/"
archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)

df = pd.read_csv(archivoAtributos)
cF = 0
cM = 0
for fila in df.itertuples():
    codigo = df["image_id"].values[fila.Index]
    sexo = df["Male"].values[fila.Index]
    archivo = os.path.join(rutaImagenes, codigo)
    if(os.path.isfile(archivo)):
        print(f"Item: {fila.Index + 1} -  Archivo: {codigo}")
        imagen = cv2.imread(archivo)
        caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in caras:
            cara = imagen[y:y+h, x:x+w]
            if(sexo==-1):
                cF+=1
                archivoCara = os.path.join(rutaFemenino, codigo)
            else:
                cM+=1
                archivoCara = os.path.join(rutaMasculino, codigo)
            cv2.imwrite(archivoCara, cara)
print(f"Total Femenino: {cF}")
print(f"Total Masculino: {cM}")
print(f"Total Caras: {cF+cM}")