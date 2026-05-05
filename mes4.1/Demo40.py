import os, cv2, joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Demo 40: App Consola Test de Raza x Carpeta")
modelo = joblib.load("Sklearn_Raza_Caras_MLP_500.pkl")
pca = joblib.load("Sklearn_Raza_Caras_PCA_MLP_500.pkl")
archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)

carpetaTest = "C:/Data/Python/2026_01_DAIAP/DataSets/UTKFace/test/"
if(os.path.isdir(carpetaTest)):
    archivos = os.listdir(carpetaTest)
    listaX=[]
    listaY=[]
    for archivo in archivos:
        campos = archivo.split("_")      
        if(len(campos)==4):
            edad = int(campos[0])
            clase = campos[2]
            if(edad>=18 and edad<=70):
                print(f"Archivo: {archivo}")
                rutaArchivo = os.path.join(carpetaTest, archivo)
                imagenGris = cv2.imread(rutaArchivo, 0)
                caras = clasificador.detectMultiScale(imagenGris, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                for(x,y,w,h) in caras:
                    cara = imagenGris[y:y+h, x:x+w]
                    nCaras = len(caras)
                    if(nCaras>0):
                        imgResize = cv2.resize(imagenGris, (100, 100))
                        imagenPlana = imgResize.flatten()
                        listaX.append(imagenPlana)
                        listaY.append(clase)
    X_test = np.array(listaX)
    y_test = np.array(listaY)
    X_test = pca.transform(X_test)

    print("Shapes de las Pruebas")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    print("Predecir las Razas")
    y_predict = modelo.predict(X_test)

    print("Evaluar el Rendimiento del Modelo")
    score = accuracy_score(y_test, y_predict)
    print("Score: ", score)

    print("Graficar la Matriz de Confusion")
    cm = confusion_matrix(y_test, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
else:
    print("Directorio Test No existe")