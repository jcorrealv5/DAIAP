import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
import torch.nn.functional as F
from modDL import CNN_2C1P
from datetime import datetime

print("Demo 15: Entrenar y Guardar un Modelo CNN de Clasificacion de Caracteres Alfabeticos")

horaInicio = datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1. Detectando el Dispositivo")
print(f"Dispositivo usado: {device}")

transform_train = transforms.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToTensor(),
    v2.RandomRotation(degrees=12),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.ElasticTransform(alpha=30.0, sigma=4.0),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
    v2.Normalize(mean=[0.5], std=[0.5])
])

print("2. Cargando el DataSet MNIST")
ruta = "C:/Data/Python/2026_01_DAIAP/DataSets/Caracteres/"
dstTrain = ImageFolder(root=ruta, transform=transform_train)
totalMuestras = len(dstTrain)
print(f"Total de Muestras: {totalMuestras}")

print("3. Configurando los HiperParametos")
sizeImage = 32
hpTamanoLote =  64
hpEpocas = 200
hpTasaAprendizaje = 0.0005

print("4. Creando el DataLoader de Caracteres")
loaderTrain = DataLoader(dstTrain, batch_size=hpTamanoLote)

caracteres = next(iter(loaderTrain))
caracter = caracteres[0]
print(f"Shape caracter: {caracter.shape}")

print("5. Crear un Modelo de Clasificacion")
modelo = CNN_2C1P(sizeImage, 1, 26).to(device)
print(f"Numero de Parametros del Modelo: {modelo.parameters()}")

print("6. Entrenar el Modelo MLP")
totalIteraciones = totalMuestras / hpTamanoLote
criterioPerdida = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=hpTasaAprendizaje)
modelo.train()
encontro = False
for i in range(hpEpocas):
    if not encontro:
        for j,(X,y) in enumerate(loaderTrain):
            X_train = X.to(device)
            y_train = y.to(device)
            y_predict = modelo(X_train)
            perdida = criterioPerdida(y_predict, y_train)
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            valor = str(round(perdida.item(),2)*100)
            print(f"Epoca {i+1} - Item: {j+1} / {totalIteraciones} - Perdida: {valor}")
            if(valor==0):
                encontro=True
                torch.save(modelo.state_dict(), "CNN_2C1P_Caracteres_" + str(valor) + ".pt")
                break

print("7. Guardar el Modelo Entrenado")
if not encontro:
    torch.save(modelo.state_dict(), "CNN_2C1P_Caracteres_" + str(valor) + ".pt")

print("8. Probar el Modelo (Predecir) y Calcular el Score")
modelo.eval()
numAciertos = 0
numMuestras = 0
with torch.no_grad():
    for j,(X,y) in enumerate(loaderTrain):
        X_test = X.to(device)
        y_test = y.to(device)
        y_predict = modelo(X_test)
        _,prediccion = y_predict.max(1)
        numAciertos += (prediccion==y_test).sum()
        numMuestras += prediccion.size(0)
score = (numAciertos / numMuestras) * 100
print("Resumen de HiperParametros y Score:")
print(f"Dispositivo usado: {device}")
print(f"Tamanio del Lote: {hpTamanoLote}")
print(f"Epocas: {hpEpocas}")
print(f"Tasa de Aprendizaje: {hpTasaAprendizaje}")
print(f"Score o Precision: {score}")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()
print(f"9. Tiempo de Procesamiento: {tiempo}")