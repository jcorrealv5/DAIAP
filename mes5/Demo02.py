from sklearn.neural_network import MLPClassifier

print("Demo 02: Perceptron MultiCapa (MLP) para Clasificacion con AND y XOR")

print("1. Crear el DataSet con los valores de la tabla logica")
X = [[1,1],[0,0],[0,1],[1,0]]
y_And = [1,0,0,0]
y_Xor = [0,0,1,1]
X_test = [[1,0], [1,1]]

print("2. Crear, entrenar y predecir con el Modelo para AND")
modeloAnd = MLPClassifier()
modeloAnd.fit(X, y_And)
y_And_predict = modeloAnd.predict(X_test)
print("X_test: ", X_test)
print("y_And_predict: ", y_And_predict)

print("3. Crear, entrenar y predecir con el Modelo para XOR")
modeloXor = MLPClassifier()
modeloXor.fit(X, y_Xor)
y_Xor_predict = modeloXor.predict(X_test)
print("X_test: ", X_test)
print("y_Xor_predict: ", y_Xor_predict)