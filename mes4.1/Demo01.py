from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("1. Preparar el DataSet con las entradas y salidas")
listaX = []
listaY = []
n = 5
for i in range(100):
    listaX.append([i+1])
    listaY.append(i+1+n)
print("listaX: ", listaX)
print("listaY: ", listaY)
X = np.array(listaX)
y = np.array(listaY)

print("2. Crear el Modelo de Regresion Lineal para Predecir")
modelo = linear_model.LinearRegression()
modelo.fit(X, y)

print("3. Preparar los datos para Predecir")
X_test = np.array([[10], [30], [100], [200], [300], [500]])
y_test = np.array([[15], [35], [105], [205], [305], [505]])
print("X_test: ", X_test)
print("y_test: ", y_test)

print("4. Realizar la Predecicion")
y_pred = modelo.predict(X_test)
print("y_pred: ", y_pred)

print("5. Evaluar el Modelo")
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print(f"Error Cuadratico Medio (MSE): {mse:.4f}")
print(f"Tasa de Precision (1 es Perefecto): {score}")
print(f"Coeficientes: {modelo.coef_}")
print(f"Intercepcion: {modelo.intercept_}")
#y = aX + b
#a = 1
#b = 5

print("6. Graficar el Modelo")
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red")
plt.show()