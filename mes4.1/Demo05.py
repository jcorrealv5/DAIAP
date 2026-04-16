from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

print("Demo 05: Regresion Lineal Simple para Predecir Precios de Casas en California")

print("1. Cargar el DataSet de Precios de Casas en California")
dst = datasets.fetch_california_housing()
print("Claves del DataSet: ", dst.keys())

print("2. Crear un DataFrame de Polar con el DataSet")
X = pd.DataFrame(data = dst["data"], columns=dst["feature_names"])
y = dst["target"]
print(f"Entradas o Features: {X.columns}")

print("3. Dividiendo la Data para Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1)
print(f"Shape X_train: {X_train.shape}")
print(f"Shape y_train: {y_train.shape}")
print(f"Shape X_test: {X_test.shape}")
print(f"Shape y_test: {y_test.shape}")

print("4. Crear el Modelo de Regresion Lineal para Predecir")
modelo = linear_model.LinearRegression()
modelo.fit(X_train, y_train)

print("5. Realizar la Predecicion")
y_pred = modelo.predict(X_test)
print("y_pred: ", y_pred)

print("6. Evaluar el Modelo")
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print(f"Error Cuadratico Medio (MSE): {mse:.4f}")
print(f"Tasa de Precision (1 es Perfecto): {score}")

print("7. Graficar el Modelo")
plt.scatter(y_test, y_pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
plt.xlabel("Precio Casa Real")
plt.ylabel("Precio Casa Predecido")
plt.title("Prediccion de Precios de Casas de California")
plt.show()