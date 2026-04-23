from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

print("Demo 13: Entrenar y Guardar un Modelo con el DataSet Predefinido MNIST-784 (28x28)")

horaInicio = datetime.now()

print("1. Cargar el DataSet usando fetch_openml")
dst = fetch_openml('mnist_784', parser="auto")
print("DataSet keys: ", dst.keys())
X = dst["data"]
y = dst["target"]
print("Shape Total X: ", X.shape)
print("Shape Total X: ", y.shape)

'''
print("Examinar el Primer Registro y Graficar")
primerX = X.iloc[0].to_numpy().reshape(28,28)
primerY = y[0]
print("Primer Digito X: ", primerX.shape)
print("Primer Digito y: ", primerY)
plt.imshow(primerX, cmap="gray")
plt.show()
'''

print("2. Dividir el DataSet en Datos de Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

print("3. Crear el Modelo con SVC")
modelo = SVC()

print("4. Entrenar el Modelo")
modelo.fit(X_train, y_train)

print("5. Guardar el Modelo en Disco")
joblib.dump(modelo, "Sklearn_MNIST784.pkl")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print(f"Tiempo de Procesamiento: {tiempo}")

print("6. Usar el Modelo para Clasificar el Digito")
y_predict = modelo.predict(X_test)

print("7. Evaluar el Rendimiento del Modelo")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)

print("8. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()