from sklearn import datasets
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Demo 04: Regresion Lineal Multiple para Predecir Diabetes")

print("1. Cargar el DataSet de Diabetes que viene en sklearn")
X, y = datasets.load_diabetes(return_X_y=True)
print("Shape X: ", X.shape)
print("Shape y: ", y.shape)

print("2. Separar los datos para entrenamiento (430) y pruebas (12)")
X_train = X[:430]
y_train = y[:430]
X_test = X[430:]
y_test = y[430:]

print("3. Crear el Modelo de Regresion Lineal para Predecir")
modelo = linear_model.LinearRegression()
modelo.fit(X_train, y_train)

print("4. Realizar la Predecicion")
y_pred = modelo.predict(X_test)
print("y_pred: ", y_pred)

print("5. Evaluar el Modelo")
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print(f"Error Cuadratico Medio (MSE): {mse:.4f}")
print(f"Tasa de Precision (1 es Perefecto): {score}")

print("6. Graficar el Modelo")
plt.scatter(y_test, y_pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
plt.xlabel("Nivel de Glucosa Real")
plt.ylabel("Nivel de Glucosa Predecido")
plt.title("Prediccion de Diabetes")
plt.hlines(y=[100,125], xmin=np.min(X_test), xmax=np.max(X_test), colors=['b', 'r'], linestyles=['-', '--'])
plt.show()