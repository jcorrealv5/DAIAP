import os, shutil
import pandas as pd

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
for archivo in archivos:
    dfFila = df[df["image_id"]==archivo]
    sexo = "M" if dfFila["Male"].values[0]==1 else "F"
    print(f"Imagen: {archivo} - Sexo: {sexo}")
    rutaOrigenImagen = os.path.join(rutaImagenes, archivo)
    rutaDestinoMasculino = os.path.join(rutaDestino + "Masculino", archivo)
    rutaDestinoFemenino = os.path.join(rutaDestino + "Femenino", archivo)
    if(sexo=="M" and cM<nMuestras):
        shutil.copy(rutaOrigenImagen, rutaDestinoMasculino)
        cM+=1
    if(sexo=="F" and cF<nMuestras):
        shutil.copy(rutaOrigenImagen, rutaDestinoFemenino)
        cF+=1
    if(cM==nMuestras and cF==nMuestras):
        break
print(f"Archivos Masculino copiados: {cM}")
print(f"fArchivos Femenino copiados: {cF}")
