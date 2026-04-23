from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Demo 12: Clasificacion Binaria de Cancer de Pulmon usando DecisionTreeClassifier")

print("1. Cargar el DataSet Predefinido de breast_cancer")
dst = load_breast_cancer()
print("Keys DataSet: ", dst.keys())
X = dst["data"]
y = dst["target"]
caracteristicas = dst["feature_names"]
etiquetas = dst["target_names"]
print("Shape Total X: ", X.shape)
print("Shape Total y: ", y.shape)
print("Caractericticas: ", caracteristicas)
print("etiquetas: ", etiquetas)

print("2. Dividir la Data en Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.1)

print("3. Crear el Modelo Clasificador con DecisionTreeClassifier")
modelo = DecisionTreeClassifier()

print("4. Entrenar el Modelo")
modelo.fit(X_train, y_train)

print("5. Clasificar los datos de pruebas si tienen tumor maligno o benigno")
y_predict = modelo.predict(X_test)

print("6. Medir el Rendimiento del Modelo Clasificador")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)

print("7. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict, labels=modelo.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot()
plt.show()