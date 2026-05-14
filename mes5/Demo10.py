import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F
from modDL import MLP
from datetime import datetime

print("Demo 10: Entrenar y Guardar un Modelo MLP de Clasificacion de Digitos MNIST")

horaInicio = datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1. Detectando el Dispositivo")
print(f"Dispositivo usado: {device}")

print("2. Cargando el DataSet MNIST")
dstTrain = MNIST(root="..\\..\\DataSets", train=True, download=True,transform=ToTensor())
totalMuestras = len(dstTrain)
print(f"Total de Muestras: {totalMuestras}")

print("3. Configurando los HiperParametos")
hpCapasOcultas = 50
hpTamanoLote =  32
hpEpocas = 10
hpTasaAprendizaje = 0.001

print("4. Creando el DataLoader del MNIST")
loaderTrain = DataLoader(dstTrain, batch_size=hpTamanoLote)

print("5. Crear un Modelo de Clasificacion")
modelo = MLP(784, hpCapasOcultas, 10).to(device)
print(f"Numero de Parametros del Modelo: {modelo.parameters()}")

print("6. Entrenar el Modelo MLP")
totalIteraciones = totalMuestras / hpTamanoLote
criterioPerdida = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=hpTasaAprendizaje)
modelo.train()
for i in range(hpEpocas):
    for j,(X,y) in enumerate(loaderTrain):
        X_train = X.to(device).reshape(X.shape[0], -1)
        y_train = y.to(device)
        y_predict = modelo(X_train)
        perdida = criterioPerdida(y_predict, y_train)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        print(f"Epoca {i+1} - Item: {j+1} / {totalIteraciones}")

print("7. Probar el Modelo (Predecir) y Calcular el Score")
modelo.eval()
numAciertos = 0
numMuestras = 0
with torch.no_grad():
    for j,(X,y) in enumerate(loaderTrain):
        X_test = X.to(device).reshape(X.shape[0], -1)
        y_test = y.to(device)
        y_predict = modelo(X_test)
        _,prediccion = y_predict.max(1)
        numAciertos += (prediccion==y_test).sum()
        numMuestras += prediccion.size(0)
score = (numAciertos / numMuestras) * 100
print("Resumen de HiperParametros y Score:")
print(f"Numero de Capas Ocultas: {hpCapasOcultas}")
print(f"Tamanio del Lote: {hpTamanoLote}")
print(f"Epocas: {hpEpocas}")
print(f"Tasa de Aprendizaje: {hpTasaAprendizaje}")
print(f"Score o Precision: {score}")

print("7. Guardar el Modelo Entrenado")
torch.save(modelo.state_dict(), "Torch_MNIST.pt")

horaFin = datetime.now()
tiempo = (horaFin - horaFin).total_seconds()
print(f"8. Tiempo de Procesamiento: {tiempo}")
