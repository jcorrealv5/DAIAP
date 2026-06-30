import torch, os
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights

class ModeloClasifica():
    def __init__(self):
        self.pesos = ResNet50_Weights.DEFAULT
        self.modelo = resnet50(weights=self.pesos)
        self.transformacion = self.pesos.transforms()
        self.modelo.eval()

    def getTransform(self):
        return self.transformacion
    
    def ClasificarImagenes(self, imagenesLote):
        rpta = []
        predicciones = self.modelo(imagenesLote)
        for prediccion in predicciones:
            idClase = prediccion.argmax().item()
            score = prediccion[idClase].item()
            clases = self.pesos.meta["categories"]
            nombreClase = clases[idClase]
            rpta.append({"Clase": nombreClase, "Score": score})
        return rpta

rutaImagenes = "C:/Data/Python/2026_01_DAIAP/Imagenes/Animales"
archivos = os.listdir(rutaImagenes)
modelo = ModeloClasifica()
imagenesLista = []
for nombreArchivo in archivos:
    archivoImagen = os.path.join(rutaImagenes, nombreArchivo)
    imagen = decode_image(archivoImagen)
    transformacion = modelo.getTransform()
    imagenTransformada = transformacion(imagen)
    imagenesLista.append(imagenTransformada)
imagenesLote = torch.stack(imagenesLista)
print(f"Shape imagenesLote: {imagenesLote.shape}")
rpta = modelo.ClasificarImagenes(imagenesLote)
c = 0
for item in rpta:
    print(f"Archivo: {archivos[c]}, Clase: {item['Clase']}, Score: {item['Score']}")
    c=c+1