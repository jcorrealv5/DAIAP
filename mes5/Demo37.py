from torchvision.models import resnet50, ResNet50_Weights
import torch
import cv2

print("Demo 37: Usando Modelos PreEntrenados de Clasificacion para 1 objeto")
archivo = r"C:\Data\Python\2026_01_DAIAP\Imagenes\Clasicos\carlos.jpg"
imagenArray = cv2.imread(archivo)
imagenArrayRGB = cv2.cvtColor(imagenArray, cv2.COLOR_BGR2RGB)
print(f"Shape Imagen Array: {imagenArrayRGB.shape}")
imagenTensor = torch.from_numpy(imagenArrayRGB).permute(2, 0, 1)
print(f"Shape Imagen Tensor: {imagenTensor.shape}")

pesos = ResNet50_Weights.DEFAULT
clases = pesos.meta["categories"]
#print(f"Clases o Categorias del DataSet: {clases}")
transformacion = pesos.transforms()
imagenTransformada = transformacion(imagenTensor)
print(f"Shape Imagen Transformada: {imagenTransformada.shape}")
imagenesEntrada = imagenTransformada.unsqueeze(0)
print(f"Shape Imagen Entrada: {imagenesEntrada.shape}")

modelo = resnet50(weights=pesos)
modelo.eval()
rpta = modelo(imagenesEntrada)

print(f"Categorias de Imagenes: {rpta.numel()}")
maxProb = torch.max(rpta)
print(f"Maxima Probabilidad: {maxProb.item()}")
categoriaIndice = torch.argmax(rpta)
print(f"Indice Categoria: {categoriaIndice.item()}")
categoriaNombre = clases[categoriaIndice]
print(f"Nombre Categoria: {categoriaNombre}")
