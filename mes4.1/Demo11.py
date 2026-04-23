from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Demo 11: Clasificar Digitos de MNIST usando DecisionTreeClassifier")
print("1. Cargar el DataSet Predefinido de MNIST")
dst = load_digits()
print("Keys DataSet: ", dst.keys())
X = dst["data"]
y = dst["target"]
print("Shape Total X: ", X.shape)
print("Shape Total y: ", y.shape)

print("2. Dividir la data para Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)
print("Shape Train X: ", X_train.shape)
print("Shape Train y: ", y_train.shape)
print("Shape Test X: ", X_test.shape)
print("Shape Test y: ", y_test.shape)

print("3. Mostrar el Primer Digito de Pruebas y Graficarlo")
print("Primer Digito Test X: ", X_test[0])
print("Primer Digito Test y: ", y_test[0])

print("4. Crear el Modelo")
modelo = DecisionTreeClassifier()

print("5. Entrenar el Modelo")
modelo.fit(X_train, y_train)

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