from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Demo 03: Perceptron MultiCapa (MLP) para Clasificacion Multiple con Digitos MNIST 8x8")

print("1. Cargar el DataSet de Digitos MNIST")
dst = load_digits()
X = dst["data"]
y = dst["target"]
print("Shape X: ", X.shape)
print("Shape y: ", y.shape)
print("Primer Digito: ", X[0])
print("Primera Etiqueta: ", y[0])

print("2. Dividir la data para entrenamiento y pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
print("Shape X_train: ", X_train.shape)
print("Shape y_train: ", y_train.shape)
print("Shape X_test: ", X_test.shape)
print("Shape y_test: ", y_test.shape)

print("3. Crear y entrenar el Modelo MLP para Clasificar Digitos")
modelo = MLPClassifier()
modelo.fit(X_train, y_train)

print("4. Probar el Modelo con la data de pruebas y mostrar el Score")
y_predict = modelo.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(f"Score del Modelo: {score}")

print("5. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()