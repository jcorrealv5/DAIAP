import cv2
import numpy as np

print("Demo 70: App de Consola para Detección de Objetos con YOLO v3 en una Imagen de Disco")

archivoImagen = ('/Users/jhon.correal/Documents/Python/Shifu/Hombre/u202423507.jpeg')
imagen = cv2.imread(archivoImagen)
ancho = imagen.shape[1]
alto = imagen.shape[0]

archivoCategorias = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/coco.names')
print("1. Mostrar las Categorias de Salida")
clases = []
with open(archivoCategorias, "r") as file:
    clases = [line.strip() for line in file.readlines()]
    print("Clases: ", clases)

print("2. Cargar Red de YOLO")
archivoPesos = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/yolov3.weights')
archivoConfig = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/yolov3.cfg')
modelo = cv2.dnn.readNet(archivoPesos, archivoConfig)

print("3. Mostrar las Capas de la Red")
nombresCapas =  modelo.getLayerNames()
print("Numero Capas Total: ", len(nombresCapas))
#print("Nombres Capas Total: ", nombresCapas)
capasSalida = [nombresCapas[i-1] for i in modelo.getUnconnectedOutLayers()]
print("Numero Capas Salida: ", len(capasSalida))
#print("Nombres Capas Salida: ", capasSalida)

print("4. Detectar Objetos con el Modelo de YOLO")
entrada = cv2.dnn.blobFromImage(imagen, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
modelo.setInput(entrada)
salida = modelo.forward(capasSalida)
print("Nro Salidas: ", len(salida))
#print("salida: ", salida)

print("5. Mostrar solo las salidas con Probabilidad o Score mayor al 50%")
class_ids = []
confidences = []
boxes = []
c = 0
for out in salida:
    for detection in out:
        c+=1
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * ancho)
            center_y = int(detection[1] * alto)
            w = int(detection[2] * ancho)
            h = int(detection[3] * alto)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
print("Cuadros detectados: ", c)
print("Cuadros detectados 50% Prob: ", len(boxes))

print("6. Eliminar los Falsos Positivos o Cuadros de sobra")
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
nDetecciones = len(indexes)
print("Cuadros detectados sin Falsos Positivos: ", nDetecciones)

print("7. Mostrar solo los cuadros detectados correctamente")
font = cv2.FONT_HERSHEY_PLAIN
colores = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
diccionario = {}
c = 0
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(clases[class_ids[i]])
        if(not label in diccionario):
            diccionario[label] = colores[c % len(colores)]
            c=c+1
        cv2.rectangle(imagen, (x, y), (x + w, y + h), diccionario[label], 2)
        cv2.putText(imagen, label, (x, y + 30), font, 3, diccionario[label], 3)
if(ancho>600 and alto>600):
    imagen = cv2.resize(imagen, (600,500))
cv2.imshow("Deteccion Objetos YOLO v3: " + str(nDetecciones), imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()