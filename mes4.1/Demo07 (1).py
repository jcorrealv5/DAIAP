from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, LinearSVR
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

print("3. Dividiendo la Data para Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1)
print(f"Shape X_train: {X_train.shape}")
print(f"Shape y_train: {y_train.shape}")
print(f"Shape X_test: {X_test.shape}")
print(f"Shape y_test: {y_test.shape}")

def entrenarProbarModelo(modelo, tipo):
    print("Entrenar Modelo: " + tipo)
    print("Tipo Data Entrenada: ", type(X_train))
    modelo.fit(X_train, y_train)

    print("Prediccion Modelo: " + tipo)
    y_pred = modelo.predict(X_test)
    print("y_pred: ", y_pred)

    print("Evaluar Modelo: " + tipo)
    mse = mean_squared_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)
    print(f"Error Cuadratico Medio (MSE): {mse:.4f}")
    print(f"Tasa de Precision (1 es Perfecto): {score}")
    print("_" * 50)

print("4. Crear Modelo de Regresion LinearSVR")
modeloLinearSVR = LinearSVR()
entrenarProbarModelo(modeloLinearSVR, "LinearSVR")

print("5. Crear Modelo de Regresion SVR Lineal")
modeloLineal = SVR(kernel="linear", max_iter=1000)
entrenarProbarModelo(modeloLineal, "SVR Lineal")

print("6. Crear Modelo de Regresion SVR Radial")
modeloRadial = SVR(kernel="rbf", max_iter=1000)
entrenarProbarModelo(modeloRadial, "SVR Radial")

print("7. Crear Modelo de Regresion SVR Polinomial")
modeloPolinomial = SVR(kernel="poly", max_iter=1000)
entrenarProbarModelo(modeloPolinomial, "SVR Polinomial")