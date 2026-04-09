import os
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Demo 03: Regresion Lineal Simple para Predecir el Tipo de Cambio (Compra o Venta) en una Moneda (Dolar o Euros)")

archivo = "TipoCambio.csv"
if(os.path.isfile(archivo)):
    moneda = input("Tipo de Moneda (D: Dolar y E:Euro): ")
    operacion = input("Tipo de Operacion (C: Compra y V:Venta): ")
    dias = int(input("Cuantos Dias Siguientes Predecira: "))
    if(moneda=="D" or moneda=="E"):
        if(operacion=="C" or operacion=="V"):
            if(operacion=="C"):
                operacion="COMPRA"
            else:
                operacion="VENTA"
            print("1. Crear el DataSet con 2 Columnas Dia y Operacion")
            df = pd.read_csv(archivo, sep=",")
            dfFiltro = df[df["MONEDA"]==moneda]            
            dfFiltro["Dia"] = range(1, len(dfFiltro) + 1)
            print(dfFiltro)
            X_train = np.array(dfFiltro[["Dia"]])
            y_train = np.array(dfFiltro[operacion])
            print("X_train: ", X_train)
            print("y_train: ", y_train)
            print("X_train max: ", X_train.max())

            print("2. Crear el Modelo de Regresion Lineal para Predecir")
            modelo = linear_model.LinearRegression()
            modelo.fit(X_train, y_train)

            print("3. Preparar la entrada para predecir: X_test")
            lista_X_test = []
            for i in range(dias):
                lista_X_test.append([X_train.max() + i + 1])
            X_test = np.array(lista_X_test)

            print("4. Realizar la Predecicion")
            y_pred = modelo.predict(X_test)
            print("y_pred: ", y_pred)

            print("6. Graficar el Modelo")
            plt.scatter(X_test, y_pred, color="blue")
            plt.plot(X_test, y_pred, color="red")
            plt.xlabel("Fecha")
            plt.ylabel(operacion)
            plt.title("Prediccion de Tipo de Cambio")
            plt.show()
        else:
            print("Operacion invalida. Solo C o V")
    else:
        print("Moneda invalida. Solo D o E")
else:
    print(f"Archivo {archivo} No existe")