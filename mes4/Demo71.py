import cv2
import numpy as np

print("Demo 71: App de Consola para Detección de Objetos con YOLO v3 en un Video en Tiempo Real")

archivoCategorias = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/coco.names')
clases = []
with open(archivoCategorias, "r") as file:
    clases = [line.strip() for line in file.readlines()]

archivoPesos = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/yolov3.weights')
archivoConfig = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/yolov3.cfg')
modelo = cv2.dnn.readNet(archivoPesos, archivoConfig)

nombresCapas =  modelo.getLayerNames()
capasSalida = [nombresCapas[i-1] for i in modelo.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if(cap.isOpened()):
    while True:
        rpta, imagen = cap.read()
        if(rpta):
            ancho = imagen.shape[1]
            alto = imagen.shape[0]

            entrada = cv2.dnn.blobFromImage(imagen, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            modelo.setInput(entrada)
            salida = modelo.forward(capasSalida)

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

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            nDetecciones = len(indexes)

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
            cv2.imshow("Video", imagen)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Video Finalizado")
else:
    print("No tiene una Camara Web Conectada")