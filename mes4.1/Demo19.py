import os, cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
from datetime import datetime

print("Demo 19: App Consola que Entrene y Guarde un Modelo de Caracteres (A-Z)")
horaInicio = datetime.now()

print("Paso 1: Crear el DataSet de Caracteres")
def crearDataSetCaracteres():
    rutaDataSet = "C:/Data/Python/2026_01_DAIAP/DataSets/Caracteres/"
    listaX = []
    listaY = []
    directorios = os.listdir(rutaDataSet)
    for directorio in directorios:
        print("_" * 100)
        print("directorio: ", directorio)
        print("_" * 100)
        archivos = os.listdir(rutaDataSet + directorio)
        for archivo in archivos:
            print(archivo)
            imagen = cv2.imread(os.path.join(rutaDataSet + directorio, archivo), 0)
            listaX.append(imagen.flatten())
            listaY.append(directorio)
    X = np.array(listaX)
    y = np.array(listaY)
    return X,y

X, y = crearDataSetCaracteres()

print("2. Dividir el DataSet en Datos de Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

print("3. Crear el Modelo con SVC")
modelo = SVC()

print("4. Entrenar el Modelo")
modelo.fit(X_train, y_train)

print("5. Guardar el Modelo en Disco")
joblib.dump(modelo, "Sklearn_Caracteres.pkl")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print(f"Tiempo de Procesamiento: {tiempo}")

print("6. Usar el Modelo para Clasificar el Caracter")
y_predict = modelo.predict(X_test)

print("7. Evaluar el Rendimiento del Modelo")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)
