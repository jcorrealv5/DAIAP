import os, cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime

print("Demo 22: Entrenar y Guardar un Modelo Binario para Detectar el Sexo")
horaInicio = datetime.now()

print("1. Crear el DataSet para el Sexo con las Caras de Hombres y Mujeres")
rutaSexo = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Sexo/"
listaX = []
listaY = []
carpetas = os.listdir(rutaSexo)
for carpeta in carpetas:
    sexo = carpeta[0]
    print("_" * 100)
    print(f"Categoria: {carpeta}")
    print("_" * 100)
    archivos = os.listdir(rutaSexo + carpeta)
    for archivo in archivos:
        print(archivo)
        rutaArchivo = os.path.join(rutaSexo + carpeta, archivo)
        imagenGris = cv2.imread(rutaArchivo, 0)
        imagenGris100 = cv2.resize(imagenGris, (100,100))
        imagenPlana = imagenGris100.flatten()
        listaX.append(imagenPlana)
        listaY.append(sexo)
X = np.array(listaX)
y = np.array(listaY)
print("Shape X: ", X.shape)
print("Shape y: ", y.shape)

print("2. Dividir el DataSet en Datos de Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

print("3. Crear el Modelo con SVC")
modelo = SVC()

print("4. Entrenar el Modelo")
modelo.fit(X_train, y_train)

print("5. Guardar el Modelo en Disco")
joblib.dump(modelo, "Sklearn_Sexo.pkl")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print(f"Tiempo de Procesamiento: {tiempo}")

print("6. Usar el Modelo para Clasificar el Sexo")
y_predict = modelo.predict(X_test)

print("7. Evaluar el Rendimiento del Modelo")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)

print("8. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()