from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.io.image import decode_image
import cv2

print("Paso 1: Crear el Modelo para Detectar Objetos")
pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos, box_score_thresh=0.9)
modelo.eval()

print("Paso 2: Leer la imagen de entrada y prepararla para el modelo")
archivoImagen = "Personas.jpg"
imagen = cv2.imread(archivoImagen)
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
categorias = list(set(prediccion["labels"]))
print(f"Categorias: {categorias}")
colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (127, 0, 0), (0, 127, 0), (0, 0, 127)]
print(f"Clases: {clases}")
for i in range(nObjetos):
    idClase = prediccion["labels"][i]
    color = colores[categorias.index(idClase)]
    nombreClase = clases[i]
    score = int(prediccion["scores"][i] * 100)
    cuadro = prediccion["boxes"][i]
    print(f"Id clase: {idClase}")
    print(f"Nombre clase: {nombreClase}")
    print(f"Score: {score}%")
    print(f"Cuadro: {cuadro}")
    x1 = int(cuadro[0].item())
    y1 = int(cuadro[1].item())
    x2 = int(cuadro[2].item())
    y2 = int(cuadro[3].item())
    print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    cv2.rectangle(imagen, rec=(x1,y1,x2,y2), color=color, thickness=3)
    cv2.putText(imagen, nombreClase + ": " + str(score), org=(x1,y1-20), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3)
cv2.imshow("Deteccion Objetos", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()