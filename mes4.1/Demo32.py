import os, cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime

print("Demo 32: Entrenar y Guardar un Modelo Muticlase para Detectar Emociones con el Rostro con MLPClassifier")
horaInicio = datetime.now()

def crearData(titulo, carpetaRaiz, nMuestras):
    listaX = []
    listaY = []
    print(titulo)
    clases = os.listdir(carpetaRaiz)
    for i,clase in enumerate(clases):
        print("_" * 100)
        print("Clase: ", clase)
        print("_" * 100)
        archivos = os.listdir(carpetaRaiz + clase)
        c = 0
        for archivo in archivos:            
            if(c<nMuestras):
                print(archivo)
                rutaArchivo = os.path.join(carpetaRaiz + clase, archivo)
                imagenGris = cv2.imread(rutaArchivo, 0)
                imagenPlana = imagenGris.flatten()
                listaX.append(imagenPlana)
                listaY.append(i)
                c+=1
            else:
                break
    X = np.array(listaX)
    y = np.array(listaY)
    return X, y

print("1. Crear la Data de Entrenamiento con 7 Emociones cada una con 2477 muestras")
carpetaRaizTrain = "C:/Data/Python/2026_01_DAIAP/DataSets/FER2013/train/"
nMuestrasTrain = 2000
X_train, y_train = crearData("Crear Data de Entrenamiento", carpetaRaizTrain, nMuestrasTrain)

print("2. Crear la Data de Pruebas con 7 Emociones cada una con 3 muestras")
carpetaRaizTest = "C:/Data/Python/2026_01_DAIAP/DataSets/FER2013/test/"
nMuestrasTest = 3
X_test, y_test = crearData("Crear Data de Pruebas", carpetaRaizTest, nMuestrasTest)

print("3. Reducir la Dimensionalidad con PCA")
pca = PCA(n_components=150, whiten=True).fit(X_train)
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
joblib.dump(modelo, "Sklearn_FER_MLP.pkl")
joblib.dump(pca, "Sklearn_FER_PCA_MLP.pkl")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print(f"Tiempo de Procesamiento: {tiempo}")

print("8. Usar el Modelo para Clasificar la Emocion")
y_predict = modelo.predict(X_test)

print("9. Evaluar el Rendimiento del Modelo")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)

print("10. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()