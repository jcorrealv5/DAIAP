from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

print("Demo 15: Entrenar y Guardar un Modelo SVC Lineal con DataSet Predefinido de Personas famosas")

horaInicio = datetime.now()

print("1. Cargar el DataSet de Personas Famosas")
dst = fetch_lfw_people()
print("Claves del DataSet: ", dst.keys())
X = dst["data"]
y = dst["target"]
etiquetas = dst["target_names"]
imagenes = dst["images"]
print("Shape X: ", X.shape)
print("Shape y: ", y.shape)
print("Shape Imgs: ", imagenes.shape)
print("Etiquetas: ", etiquetas)

'''
print("Examinar la Primera Muestra")
primerX = X[0]
primerY = y[0]
primerImg = imagenes[0]
nombre = etiquetas[primerY]
print("Shape Primera Entrada X: ", primerX.shape)
print("Primera Salida y: ", primerY)
print("Shape Primera Imagen X: ", primerImg.shape)
plt.imshow(primerImg, cmap="gray")
plt.title(str(primerY) + " = " + nombre)
plt.show()
'''

print("2. Dividir el DataSet en Datos de Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

print("3. Crear el Modelo con SVC")
modelo = SVC()

print("4. Entrenar el Modelo")
modelo.fit(X_train, y_train)

print("5. Guardar el Modelo en Disco")
joblib.dump(modelo, "Sklearn_LFW_People.pkl")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print(f"Tiempo de Procesamiento: {tiempo}")

print("6. Usar el Modelo para Clasificar el Digito")
y_predict = modelo.predict(X_test)

print("7. Evaluar el Rendimiento del Modelo")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)