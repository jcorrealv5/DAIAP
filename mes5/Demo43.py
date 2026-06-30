import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.io.image import decode_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

print("Paso 1: Crear el Modelo para Detectar Objetos")
pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos, box_score_thresh=0.9)
modelo.eval()

print("Paso 2: Leer la imagen de entrada y prepararla para el modelo")
archivoImagen = "Cuarto_Gente.jpg"
imagenTensor = decode_image(archivoImagen)
print(f"Shape Imagen Tensor: {imagenTensor.shape}")
transformacion = pesos.transforms()
imagenTransformada = transformacion(imagenTensor)
print(f"Shape Imagen Transformada: {imagenTransformada.shape}")
imagenesLote = imagenTransformada.unsqueeze(0)
print(f"Shape Imagenes Lote: {imagenesLote.shape}")

print("Paso 3: Ejecutar la Inferencia osea la Deteccion de Objetos")
prediccion = modelo(imagenesLote)[0]
nObjetos = prediccion["labels"].numel()
clases = [pesos.meta["categories"][i] for i in prediccion["labels"]]
colores = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta", "lime", "teal", "lavender", "brown", "beige", "maroon", "mint", "olive", "coral", "navy", "grey"]
listaColores = [colores[int(clase) % len(colores)] for clase in prediccion["labels"]]
imagenCuadros = draw_bounding_boxes(imagenTransformada, boxes=prediccion["boxes"],labels=clases,colors=listaColores,width=4, font_size=50)
imagen = to_pil_image(imagenCuadros.detach())
imagen.show()