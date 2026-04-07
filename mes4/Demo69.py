import cv2

print("Demo 69: App de Consola para Detección de Marcas Faciales en un Video en Tiempo Real")

archivoHaar = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/haarcascade_frontalface_default.xml')
clasificador = cv2.CascadeClassifier(archivoHaar)

archivoModeloLBF = ('/Users/jhon.correal/Documents/Python/Shifu/Archivos/lbfmodel.yaml')
lbf = cv2.face.createFacemarkLBF()
lbf.loadModel(archivoModeloLBF)

video = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if(video.isOpened()):
    while(True):
        rpta, imagen = video.read()
        if(rpta):
            caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            nCaras = len(caras)
            if(nCaras>0):
                _, marcasFaciales = lbf.fit(imagen, caras)            
                for marca in marcasFaciales:
                    for i,(x,y) in enumerate(marca[0]):
                        #Pintar los Bordes del Rostro de Color Verde
                        if(i>=0 and i<=26):
                            cv2.circle(imagen, (int(x), int(y)), 1, (0, 255, 0), 2)
                        #Pintar la Nariz de Color Azul
                        if(i>=27 and i<=35):
                            cv2.circle(imagen, (int(x), int(y)), 1, (255, 0, 0), 2)
                        #Pintar los Ojos de Color Celeste
                        if(i>=36 and i<=47):
                            cv2.circle(imagen, (int(x), int(y)), 1, (255, 255, 0), 2)
                        #Pintar los Labios de Color Rojo
                        if(i>=48 and i<=67):
                            cv2.circle(imagen, (int(x), int(y)), 1, (0, 0, 255), 2)            
            imagen = cv2.resize(imagen, (600,500))
            cv2.imshow("Detectar Marcas Faciales en Tiempo Real", imagen)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
    video.release()
    cv2.destroyAllWindows()
else:
    print("Camara No esta activa")