from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Demo 10: Clasificacion de una Flor Iris usando SVM")

print("1. Cargar el DataSet de Iris")
dst = load_iris()
print("Keys: ", dst.keys())
X = dst["data"]
y = dst["target"]
caracteristicas = dst["feature_names"]
etiquetas = dst["target_names"]
print("Shape Total X: ", X.shape)
print("Shape Total y: ", y.shape)
print("Caracteristicas: ", caracteristicas)
print("Etiquetas: ", etiquetas)
primerEntrada = X[0]
primerSalida = y[0]
print("Primera Entrada: ", primerEntrada)
print("Primera Salida: ", primerSalida)

print("2. Crear un Modelo de Clasificacion con SVC")
modelo = svm.SVC()

print("3. Separar la Data para Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
print("Shape Train X: ", X_train.shape)
print("Shape Train y: ", y_train.shape)
print("Shape Test X: ", X_test.shape)
print("Shape Test y: ", y_test.shape)

print("4. Entrenar el modelo")
modelo.fit(X_train, y_train)

print("5. Usar el Modelo para Clasificar")
y_predict = modelo.predict(X_test)
print("Valores Reales: ", y_test)
print("Valores Predecidos: ", y_predict)
aciertos = [y_test==y_predict]
print("Aciertos: ", aciertos)
score1 = np.count_nonzero(aciertos) / len(aciertos[0])
print("Score1: ", score1)

print("6. Evaluar el Rendimiento del Modelo")
score2 = accuracy_score(y_test, y_predict)
print("Score2: ", score2)

print("7. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict, labels=modelo.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot()
plt.show()