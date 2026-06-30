from torchvision.models import resnet50, ResNet50_Weights
import torch, cv2, os, shutil

print("Demo 38: Usando Modelos PreEntrenados de Clasificacion para 1 objeto")
rutaOrigen = "C:/Data/Python/2026_01_DAIAP/Imagenes/Animales/"
rutaDestino = "C:/Data/Python/2026_01_DAIAP/DataSets/Animales/"

pesos = ResNet50_Weights.DEFAULT
clases = pesos.meta["categories"]
transformacion = pesos.transforms()
modelo = resnet50(weights=pesos)
modelo.eval()

print("Iniciando proceso de Clasificacion...")
if(os.path.isdir(rutaOrigen)):
    archivos = os.listdir(rutaOrigen)
    for i,nombreArchivo in enumerate(archivos):
        print(f"Archivo {i+1}: {nombreArchivo}")
        #1. Preparar la imagen a predecir
        archivoOrigen = os.path.join(rutaOrigen, nombreArchivo)
        imagenArray = cv2.imread(archivoOrigen)
        imagenArrayRGB = cv2.cvtColor(imagenArray, cv2.COLOR_BGR2RGB)
        imagenTensor = torch.from_numpy(imagenArrayRGB).permute(2, 0, 1)
        imagenTransformada = transformacion(imagenTensor)
        imagenesEntrada = imagenTransformada.unsqueeze(0)
        #2. Predecir
        rpta = modelo(imagenesEntrada)
        maxProb = torch.max(rpta)
        categoriaIndice = torch.argmax(rpta)
        categoriaNombre = clases[categoriaIndice]
        print(f"Categoria: {categoriaNombre}")
        #3. Copiar el archivo al destino
        carpetaDestino = rutaDestino + categoriaNombre
        if(not os.path.isdir(carpetaDestino)):
            os.makedirs(carpetaDestino)
        archivoDestino = os.path.join(carpetaDestino, nombreArchivo)
        shutil.copyfile(archivoOrigen, archivoDestino)
    print("Se proceso todas las imagenes")
else:
    print("No existe el directorio de Animales a Procesar")