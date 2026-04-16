from sklearn import datasets
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Demo 06: Regresion Lineal Multiple para Predecir Diabetes")

print("1. Cargar el DataSet de Diabetes que viene en sklearn")
X, y = datasets.load_diabetes(return_X_y=True)
print("Shape X: ", X.shape)
print("Shape y: ", y.shape)

print("2. Separar los datos para entrenamiento (430) y pruebas (12)")
X_train = X[:430]
y_train = y[:430]
X_test = X[430:]
y_test = y[430:]

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

print("3. Crear Modelo de Regresion Lineal")
modeloLineal = SVR(kernel="linear", C=100, gamma="auto")
entrenarProbarModelo(modeloLineal, "Lineal")

modeloRadial = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
entrenarProbarModelo(modeloRadial, "Radial")

modeloPolinomial = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
entrenarProbarModelo(modeloPolinomial, "Polinomial")