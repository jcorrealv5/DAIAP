import torch
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights

print("Paso 1: Leer la Imagen a Clasificar")
archivoImagen = r"C:\Data\Python\2026_01_DAIAP\Imagenes\Clasicos\Zebra.jpg"
imagen = decode_image(archivoImagen)
print(f"Shape Imagen Tensor: {imagen.shape}")

print("Paso 2: Crear el Modelo usando ResNet50")
pesos = ResNet50_Weights.DEFAULT
modelo = resnet50(weights=pesos)
modelo.eval()
#print(f"Modelo: {modelo}")

print("Paso 3: Preparar la Imagen que es la Entrada del Modelo de Vision")
transformacion = pesos.transforms()
imagenTransformada = transformacion(imagen)
print(f"Shape Imagen Transformada: {imagenTransformada.shape}")
imagenLote = imagenTransformada.unsqueeze(0)
print(f"Shape Imagen Lote: {imagenLote.shape}")

print("Paso 4: Realizar la inferencia o clasificar la imagen de entrada")
prediccion = modelo(imagenLote).squeeze(0).softmax(0)
#print(f"Prediccion: {prediccion}")
idClase = prediccion.argmax().item()
print(f"Id Clase: {idClase}")
score = prediccion[idClase].item()
print(f"Score: {score * 100}")
clases = pesos.meta["categories"]
#print(f"Clases: {clases}")
nombreClase = clases[idClase]
print(f"Nombre Clase: {nombreClase}")