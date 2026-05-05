import os, cv2
import joblib

print("Demo 30: Detectar el sexo y lentes en Tiempo Real")

cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    modelo = joblib.load("Sklearn_Raza_Caras_MLP_500.pkl")
    pca = joblib.load("Sklearn_Raza_Caras_PCA_MLP_500.pkl")
    archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
    clasificador = cv2.CascadeClassifier(archivoHaar)
    razas = ["Blanca", "Negra", "Asiatica", "India", "Otros"]
    colores = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
    while True:
        rpta, imagen = cap.read()
        if(rpta):
            imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            grosor = 3
            caras = clasificador.detectMultiScale(imagenGris, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in caras:
                cara = imagenGris[y:y+h, x:x+w]
                imgResize = cv2.resize(cara, (100, 100))
                imagenPlana = imgResize.flatten()
                X_test = imagenPlana.reshape(1, -1)
                X_test = pca.transform(X_test)
                y_predecido = modelo.predict(X_test)
                indice = int(y_predecido[0])
                nombre = razas[indice]
                color = colores[indice]
                cv2.rectangle(imagen, rec=(x,y,w,h), color=color, thickness=grosor)
                cv2.putText(imagen, nombre, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=grosor)
            cv2.imshow("Raza", imagen)
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