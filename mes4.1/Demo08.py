from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor

print("Demo 08: Seleccion del Mejor Algoritmo de Regresion para el DataSet de Diabetes")

print("Paso 1: Crear el DataSet")
dst = load_diabetes()
print("Keys: ", dst.keys())
X = dst["data"]
y = dst["target"]
caracteristicas = dst["feature_names"]

print("Shape X: ", X.shape)
print("Shape y: ", y.shape)
print("caracteristicas: ", caracteristicas)
print("Entrada Primer Registro: ", X[0])
print("Salida Primer Registro: ", y[0])

print("Paso 2: Dividir el DataSet en Data de Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

estimadores = [("dtr", DecisionTreeRegressor()), ("svr", SVR())]

nombresModelos = ["LinearRegression", "RidgeCV", "SVR", "DecisionTreeRegressor",
"AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
"HistGradientBoostingRegressor", "RandomForestRegressor", 
"VotingRegressor"]
#, "StackingRegressor"

modelos = [LinearRegression(), RidgeCV(), SVR(kernel="linear"), DecisionTreeRegressor(),
AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(),
HistGradientBoostingRegressor(), RandomForestRegressor(), 
VotingRegressor(estimators=estimadores)
] 
#,StackingRegressor(estimators=estimadores, final_estimator=RandomForestRegressor())

scores = []
y_predicts = []
for i,modelo in enumerate(modelos):
    print("Crear Modelo " + str(i+1) + ": " + nombresModelos[i])
    #print("Paso 4: Entrenar el Modelo con la Data de Entrenamiento")
    modelo.fit(X_train, y_train)

    #print("Paso 5: Probar el Modelo con la Data de Pruebas")
    y_predict = modelo.predict(X_test[:100])
    y_predicts.append(y_predict)

    #print("Paso 6: Medir la Precision del Modelo en las Pruebas")
    score = r2_score(y_test[:100], y_predict)
    scores.append(score)
    print("Score del Modelo: ", score)

indice = np.argmax(scores)
modeloGanador = modelos[indice]
scoreGanador = scores[indice]
y_predictGanador = y_predicts[indice]
nombreGanador = nombresModelos[indice]

print("Paso 7: Graficar lo Predecido Vs lo Real")
x = np.linspace(0, np.max(y_test[:100]), len(y_test[:100]), dtype=float)
plt.plot(x, y_test[:100], color="red", label="Precio Real")
plt.plot(x, y_predictGanador, color="blue", label="Precio Pred")
plt.legend()
plt.title("Ganador: " + nombreGanador + " - Score: " + str(scoreGanador))
plt.show()