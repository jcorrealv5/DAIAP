import os, cv2
import joblib

print("Demo 30: Detectar el sexo y lentes en Tiempo Real")

cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    modeloSexo = joblib.load("Sklearn_Sexo.pkl")
    modeloLentes = joblib.load("Sklearn_Lentes.pkl")
    archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
    clasificador = cv2.CascadeClassifier(archivoHaar)
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
                y_sexo_predecido = modeloSexo.predict(X_test)
                y_lentes_predecido = modeloLentes.predict(X_test)
                sexo = y_sexo_predecido[0]
                lentes = y_lentes_predecido[0]
                if(sexo=="F" and lentes=="C"):
                    nombre = "Mujer con Lentes"
                    color=(255,0,0)
                if(sexo=="M" and lentes=="C"):
                    nombre = "Hombre con Lentes"
                    color=(0,0,255)
                if(sexo=="F" and lentes=="S"):
                    nombre = "Mujer sin Lentes"
                    color=(0,255,0)
                if(sexo=="M" and lentes=="S"):
                    nombre = "Hombre sin Lentes"
                    color=(0,255,255)
                cv2.rectangle(imagen, rec=(x,y,w,h), color=color, thickness=grosor)
                cv2.putText(imagen, nombre, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=grosor)
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