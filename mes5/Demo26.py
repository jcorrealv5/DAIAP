import pandas as pd
import cv2, os

rutaImagenes = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/img_align_celeba/"
archivoAtributos = r"C:\Data\Python\2026_01_DAIAP\DataSets\CelebA\list_attr_celeba.csv"
rutaCalvo = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Pelos/Calvo/"
rutaPelo = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Pelos/Pelos/"
archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)

df = pd.read_csv(archivoAtributos)
dfCalvos = df[df['Bald'] == 1]
dfPelos = df[df['Bald'] == -1]
cC = 0
cP = 0
for i,fila in enumerate(dfCalvos.itertuples()):
    codigo = dfCalvos["image_id"].values[i]
    archivo = os.path.join(rutaImagenes, codigo)
    if(os.path.isfile(archivo)):
        print(f"Calvo: {i} -  Archivo: {codigo}")
        imagen = cv2.imread(archivo)
        caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in caras:
            cara = imagen[y:y+h, x:x+w]
        if(len(caras)>0):
            cC+=1
            archivoCara = os.path.join(rutaCalvo, codigo)
            cv2.imwrite(archivoCara, cara)

for i,fila in enumerate(dfPelos.itertuples()):
    if(cP<cC):
        codigo = dfPelos["image_id"].values[i]
        archivo = os.path.join(rutaImagenes, codigo)
        if(os.path.isfile(archivo)):
            print(f"Pelo: {i} -  Archivo: {codigo}")
            imagen = cv2.imread(archivo)
            caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in caras:
                cara = imagen[y:y+h, x:x+w]
            if(len(caras)>0):
                cP+=1
                archivoCara = os.path.join(rutaPelo, codigo)
                cv2.imwrite(archivoCara, cara)
    else:
        break

print(f"Total Calvos: {cC}")
print(f"Total Pelos: {cP}")
print(f"Total Caras: {cC+cP}")