import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
import torch.nn.functional as F
from modDL import DatasetBinCat,CNN_6C3P3FC2S
from datetime import datetime

print("Demo 33: Entrenar y Guardar un Modelo CNN de Clasificacion de Sexo y Raza")

horaInicio = datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1. Detectando el Dispositivo")
print(f"Dispositivo usado: {device}")

transform_train = transforms.Compose([
    v2.Resize((156,156)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])

print("2. Cargando el DataSet UTKFace para Train")
rutaTrain = "C:/Data/Python/2026_01_DAIAP/DataSets/UTKFace/train"
dstTrain = DatasetBinCat(ruta=rutaTrain, transform=transform_train,separador="_")
totalTrain = len(dstTrain)
print(f"Total de Muestras Train: {totalTrain}")

print("3. Cargando el DataSet UTKFace para Test")
rutaTest = "C:/Data/Python/2026_01_DAIAP/DataSets/UTKFace/test"
dstTest = DatasetBinCat(ruta=rutaTest, transform=transform_train,separador="_")
totalTest = len(dstTest)
print(f"Total de Muestras Test: {totalTest}")

print("4. Configurando los HiperParametos")
sizeImage = 156
hpTamanoLote =  64
hpEpocas = 30
hpTasaAprendizaje = 0.001

print("5. Creando los DataLoaders de Entrenamiento y Pruebas")
loaderTrain = DataLoader(dstTrain, batch_size=hpTamanoLote, drop_last=True, shuffle=True)
loaderTest = DataLoader(dstTest, batch_size=hpTamanoLote, drop_last=True, shuffle=False)

print("6. Crear un Modelo de Clasificacion")
modelo = CNN_6C3P3FC2S(5, 156).to(device)

print("7. Entrenar el Modelo CNN con 6 Convoluciones, 3 Pooling, 3 Capas Densas y 2 Salidas")
totalIteraciones = int(totalTrain / hpTamanoLote)
criterioBin = nn.BCEWithLogitsLoss()
criterioCat = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=hpTasaAprendizaje, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', patience=5)
modelo.train()
mejor_perdida = float('inf')
encontro = False
for i in range(hpEpocas):
    perdida_acumulada = 0
    if not encontro:
        for j,(X,y_sexo,y_raza) in enumerate(loaderTrain):
            X_train = X.to(device)
            y_pred_sexo, y_pred_raza = modelo(X_train)
            lossSexo = criterioBin(y_pred_sexo, y_sexo.to(device).reshape(hpTamanoLote,1).float())
            lossRaza = criterioCat(y_pred_raza, y_raza.to(device))
            lossTotal = lossSexo + lossRaza
            perdida_acumulada += lossTotal.item()
            optimizador.zero_grad()
            lossTotal.backward()
            optimizador.step()                        
            print(f"Epoca {i+1}/{hpEpocas} - Item {j}/{totalIteraciones} - Perdida: {lossTotal.item():.4f}")
        perdida_promedio = perdida_acumulada / totalIteraciones
        scheduler.step(perdida_promedio)
        if perdida_promedio < mejor_perdida:
            mejor_perdida = perdida_promedio
            torch.save(modelo.state_dict(), f"CNN_6C3P3FC2S_SexoRaza_{perdida_promedio:.4f}.pt")

print("8. Probar el Modelo (Predecir) y Calcular el Score")
modelo.eval()
num_correct = 0
num_samples = 0
with torch.no_grad():
    for j,(X,y_sexo,y_raza) in enumerate(loaderTest):
        X_test = X.to(device)
        y_sexo = y_sexo.to(device)
        y_raza = y_raza.to(device)
        y_pred_sexo, y_pred_raza = modelo(X_test)
        predicSexo = (torch.sigmoid(y_pred_sexo) > 0.5).squeeze().long()
        _, predicRaza = y_pred_raza.max(1)
        num_correct += (predicSexo == y_sexo.to(device) and predicRaza == y_raza.to(device)).sum()
        num_samples += predicSexo.size(0)
score = (num_correct / num_samples) * 100
print("Resumen de HiperParametros y Score:")
print(f"Dispositivo usado: {device}")
print(f"Tamanio del Lote: {hpTamanoLote}")
print(f"Epocas: {hpEpocas}")
print(f"Tasa de Aprendizaje: {hpTasaAprendizaje}")
print(f"Score o Precision: {score}")

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()
print(f"8. Tiempo de Procesamiento: {tiempo}")