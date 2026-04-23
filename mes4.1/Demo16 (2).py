from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import loguniform
from tqdm import tqdm
import numpy as np
#from sklearn.ensemble import RandomForestClassifier

print("Demo 16: Entrenar y Guardar un Modelo SVC Radial con DataSet Predefinido de Personas famosas")

horaInicio = datetime.now()

print("1. Cargar el DataSet de Personas Famosas")
dst = fetch_lfw_people(min_faces_per_person=70)
print("Claves del DataSet: ", dst.keys())
X = dst["data"]
y = dst["target"]
n_Caracteristicas = X.shape[1]
etiquetas = dst["target_names"]
print("Shape X: ", X.shape)
print("Shape y: ", y.shape)
print("Nro de Caracteristicas: ", n_Caracteristicas)
print("Etiquetas: ", etiquetas)
print("Nro Etiquetas: ", len(etiquetas))
with open("lfw_people.txt", "w") as file:
    for etiqueta in etiquetas:
        file.write(etiqueta + "\n")

print("2. Dividir el DataSet en Datos de Entrenamiento y Pruebas")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
print("Shape X_train: ", X_train.shape)
print("Shape y_train: ", y_train.shape)
print("Shape X_test: ", X_test.shape)
print("Shape y_test: ", y_test.shape)
print("Antes de Normalizar: ", X_train[0])

print("3. Normalizar los datos de entrada")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Despues de Normalizar: ", X_train[0])

print("4. Aplicar PCA a la entrada para reducir la dimensionalidad")
n_components = 150
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Shape X_train_pca: ", X_train_pca.shape)
print("Shape X_test_pca: ", X_test_pca.shape)

print("5. Crear el Modelo Radial con SVC")
'''
param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
modelo = RandomizedSearchCV(SVC(class_weight="balanced"),
    param_grid, n_iter=6, n_jobs=-1,  verbose=2, cv=3
)
'''
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}
modelo = RandomizedSearchCV(
    SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
)
print("6. Entrenar el Modelo")
modelo.fit(X_train_pca, y_train)

print("7. Guardar el Modelo en Disco")
joblib.dump(modelo, "Sklearn_LFW_People_PCA.pkl")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print(f"Tiempo de Procesamiento: {tiempo}")

print("8. Usar el Modelo para Clasificar Rostros")
y_predict = modelo.predict(X_test_pca)
print("Shape y_test: ", y_test.shape)
print("Shape y_predict: ", y_predict.shape)

print("9. Evaluar el Rendimiento del Modelo")
score = accuracy_score(y_test, y_predict)
print("Score: ", score)

print("10. Graficar la Matriz de Confusion")
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()