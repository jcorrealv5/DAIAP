import os, cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime

print("Demo 37: Entrenar y Guardar un Modelo Multiclase para Detectar Raza con el Rostro con MLPClassifier")
horaInicio = datetime.now()
archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)

def crearData(titulo, carpetaRaiz, nMuestras):
    listaX = []
    listaY = []
    print(titulo)
    archivos = os.listdir(carpetaRaiz)
    c = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
    for archivo in archivos:
        campos = archivo.split("_")      
        if(len(campos)==4):
            edad = int(campos[0])
            clase = campos[2]
            if(edad>=18 and edad<=70 and c[clase]<nMuestras):                
                print(archivo)
                rutaArchivo = os.path.join(carpetaRaiz, archivo)
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
                        c[clase]+=1
        else:
            break
    X = np.array(listaX)
    y = np.array(listaY)
    print("Contadores: ", c)
    return X, y

print("1. Crear la Data de Entrenamiento con 5 Razas cada una con 2000 muestras")
carpetaRaizTrain = "C:/Data/Python/2026_01_DAIAP/DataSets/UTKFace/train/"
nMuestrasTrain = 2000
X_train, y_train = crearData("Crear Data de Entrenamiento", carpetaRaizTrain, nMuestrasTrain)

print("2. Crear la Data de Pruebas con 5 Emociones cada una con 3 muestras")
carpetaRaizTest = "C:/Data/Python/2026_01_DAIAP/DataSets/UTKFace/test/"
nMuestrasTest = 4
X_test, y_test = crearData("Crear Data de Pruebas", carpetaRaizTest, nMuestrasTest)

print("3. Reducir la Dimensionalidad con PCA")
pca = PCA(n_components=500, whiten=True).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print("4. Mostrar los Shapes del DataSet")
print("Shape X Train: ", X_train.shape)
print("Shape y Train: ", y_train.shape)
print("Shape X Test: ", X_test.shape)
print("Shape y Test: ", y_test.shape)

print("5. Crear el Modelo con MLP")
modelo = MLPClassifier(hidden_layer_sizes=100,random_state=1, max_iter=300)

print("6. Entrenar el Modelo")
modelo.fit(X_train, y_train)

print("7. Guardar el Modelo en Disco")
joblib.dump(modelo, "Sklearn_Raza_Caras_MLP_500.pkl")
joblib.dump(pca, "Sklearn_Raza_Caras_PCA_MLP_500.pkl")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print(f"Tiempo de Procesamiento: {tiempo}")

print("8. Usar el Modelo para Clasificar la Raza")
y_predict = modelo.predict(X_test)

print("9. Evaluar el Rendimiento del Modelo")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)

print("10. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()