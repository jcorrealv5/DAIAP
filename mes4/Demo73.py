import cv2
from ultralytics import YOLO

archivoYolo = r"C:\Data\Python\2026_01_DAIAP\Archivos\yolo26n.pt"
print("Demo 73: App de Consola para Detección de Objetos con YOLO v26 en una Imagen de Disco")
modelo = YOLO(archivoYolo)

archivoImagen = r"C:\Data\Python\2026_01_DAIAP\Imagenes\Clasicos\Jorge_Profesor_Pedro.jpeg"
salida = modelo(archivoImagen)
for rpta in salida:
    imagenDetectada = rpta.plot()
    if(imagenDetectada.shape[0]>600 and imagenDetectada.shape[1]>600):
        imagenDetectada = cv2.resize(imagenDetectada, (600,500))
    cv2.imshow("Yolo v26", imagenDetectada)
    cv2.waitKey(0)
cv2.destroyAllWindows()