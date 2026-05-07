import numpy as np
from sklearn.neural_network import MLPRegressor

print("Demo 01: Perceptron MultiCapa (MLP) para Suma")

print("1. Crear el DataSet con los numeros")
X = np.random.randint(100, size=(1000,2))
y = np.sum(X, axis=1)
print("X: ", X)
print("y: ", y)

print("2. Crear el Modelo MCP")
modelo = MLPRegressor(solver='lbfgs', alpha=1e-5, max_iter=300, hidden_layer_sizes=(6,), random_state=1)

print("3. Entrenar el Modelo")
modelo.fit(X, y)

print("4. Predecir para Nuevos datos")
X_test = [[5,7],[4,8],[9,15],[10,50],[100,400]]
print("X_test: ", X_test)
y_pred = modelo.predict(X_test)
print("y_pred: ", y_pred)