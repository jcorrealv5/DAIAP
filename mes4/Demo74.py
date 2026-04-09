import cv2
from ultralytics import YOLO

archivoYolo = r"C:\Data\Python\2026_01_DAIAP\Archivos\yolo26n.pt"
print("Demo 74: App de Consola para Detección de Objetos con YOLO v26 en un Video en Tiempo Real")
modelo = YOLO(archivoYolo)

cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    while True:
        rpta, imagen = cap.read()
        if(rpta):
            salida = modelo(imagen)
            for rpta in salida:
                imagenDetectada = rpta.plot()
                cv2.imshow("Video Yolo v26", imagenDetectada)
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