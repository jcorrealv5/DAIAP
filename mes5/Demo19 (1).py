import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
import torch.nn.functional as F
from modDL import CNN_4C4P
from datetime import datetime

print("Demo 19: Entrenar y Guardar un Modelo CNN de Clasificacion de Sexo")

horaInicio = datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1. Detectando el Dispositivo")
print(f"Dispositivo usado: {device}")

transform_train = transforms.Compose([
    v2.Resize((142,142)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(10),
    v2.Grayscale(num_output_channels=1),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])

print("2. Cargando el DataSet CelebA")
ruta = "C:/Data/Python/2026_01_DAIAP/DataSets/CelebA/Sexo"
dstTrain = ImageFolder(root=ruta, transform=transform_train)
totalMuestras = len(dstTrain)
print(f"Total de Muestras: {totalMuestras}")

print("3. Configurando los HiperParametos")
sizeImage = 142
hpTamanoLote =  64
hpEpocas = 30
hpTasaAprendizaje = 0.001

print("4. Creando el DataLoader CelebA")
loaderTrain = DataLoader(dstTrain, batch_size=hpTamanoLote, drop_last=True)

caras = next(iter(loaderTrain))
cara = caras[0]
print(f"Shape cara: {cara.shape}")

print("5. Crear un Modelo de Clasificacion")
modelo = CNN_4C4P(sizeImage, 1, 1).to(device)

print("6. Entrenar el Modelo CNN con 4 Convoluciones y 4 Pooling")
totalIteraciones = int(totalMuestras / hpTamanoLote)
criterioPerdida = nn.BCEWithLogitsLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=hpTasaAprendizaje, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', patience=5)
modelo.train()
mejor_perdida = float('inf')
encontro = False
for i in range(hpEpocas):
    perdida_acumulada = 0
    if not encontro:
        for j,(X,y) in enumerate(loaderTrain):
            X_train = X.to(device)
            y_train = y.to(device).reshape(hpTamanoLote,1).float()
            y_predict = modelo(X_train).float()
            perdida = criterioPerdida(y_predict, y_train)
            perdida_acumulada += perdida.item()
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            perdida_promedio = perdida_acumulada / totalIteraciones
            scheduler.step(perdida_promedio)
            print(f"Epoca {i+1}/{hpEpocas} - Item {j}/{totalIteraciones} - Perdida: {perdida.item():.4f}")
        if perdida_promedio < mejor_perdida:
            mejor_perdida = perdida_promedio
            torch.save(modelo.state_dict(), f"CNN_3C3P_Sexo_{perdida_promedio:.4f}.pt")

print("7. Probar el Modelo (Predecir) y Calcular el Score")
modelo.eval()
numAciertos = 0
numMuestras = 0
with torch.no_grad():
    for j,(X,y) in enumerate(loaderTrain):
        X_test = X.to(device)
        y_test = y.to(device)
        y_predict = modelo(X_test)
        prediccion = (torch.sigmoid(y_predict) > 0.5).float().squeeze().long()
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
print(f"8. Tiempo de Procesamiento: {tiempo}")