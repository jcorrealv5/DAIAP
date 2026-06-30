import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
import torch.nn.functional as F
from modDL import DatasetCelebABin8,CNN_6C3P3FC8S
from datetime import datetime

print("Demo 35: Entrenar y Guardar un Modelo CNN de Clasificacion de 8 Atributos")

horaInicio = datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1. Detectando el Dispositivo")
print(f"Dispositivo usado: {device}")

transform_train = transforms.Compose([
    v2.Resize((156,156)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])

print("2. Cargando el DataSet CelebA para Train: 0 - 10000")
dstTrain = DatasetCelebABin8(inicio=0, fin=100000, transform=transform_train)
totalTrain = len(dstTrain)
print(f"Total de Muestras Train: {totalTrain}")

print("3. Cargando el DataSet CelebA para Test: 10001 - 11000")
dstTest = DatasetCelebABin8(inicio=100001, fin=105000, transform=transform_train)
totalTest = len(dstTest)
print(f"Total de Muestras Test: {totalTest}")

print("4. Configurando los HiperParametos")
sizeImage = 156
hpTamanoLote =  64
hpEpocas = 1
hpTasaAprendizaje = 0.001

print("5. Creando los DataLoaders de Entrenamiento y Pruebas")
loaderTrain = DataLoader(dstTrain, batch_size=hpTamanoLote, drop_last=True, shuffle=True)
loaderTest = DataLoader(dstTest, batch_size=hpTamanoLote, drop_last=True, shuffle=False)

'''
totalPersonas = 0
totalHombres = 0
totalMujeres = 0
with torch.no_grad():
    for j,(X,y_calvo, y_cabelloNegro, y_gordito, y_gafas, y_masculino, y_bigote, y_sombrero, y_joven) in enumerate(loaderTrain):
        totalPersonas += y_masculino.numel()
        totalHombres += y_masculino.sum()
totalMujeres = totalPersonas - totalHombres
print(f"Total de Hombres: {totalHombres}")
print(f"Total de Mujeres: {totalMujeres}")
'''

print("6. Crear un Modelo de Clasificacion")
modelo = CNN_6C3P3FC8S(156).to(device)

print("7. Entrenar el Modelo CNN con 6 Convoluciones, 3 Pooling, 3 Capas Densas y 8 Salidas")
totalIteraciones = int(totalTrain / hpTamanoLote)
criterioBin = nn.BCEWithLogitsLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=hpTasaAprendizaje, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', patience=5)
modelo.train()
mejor_perdida = float('inf')
encontro = False
for i in range(hpEpocas):
    perdida_acumulada = 0
    if not encontro:
        for j,(X,y_calvo, y_cabelloNegro, y_gordito, y_gafas, y_masculino, y_bigote, y_sombrero, y_joven) in enumerate(loaderTrain):
            X_train = X.to(device)
            y_calvo = y_calvo.to(device).reshape(hpTamanoLote,1).float()
            y_cabelloNegro = y_cabelloNegro.to(device).reshape(hpTamanoLote,1).float()
            y_gordito = y_gordito.to(device).reshape(hpTamanoLote,1).float()
            y_gafas = y_gafas.to(device).reshape(hpTamanoLote,1).float()
            y_masculino = y_masculino.to(device).reshape(hpTamanoLote,1).float()
            y_bigote = y_bigote.to(device).reshape(hpTamanoLote,1).float()
            y_sombrero = y_sombrero.to(device).reshape(hpTamanoLote,1).float()
            y_joven = y_joven.to(device).reshape(hpTamanoLote,1).float()
            y_pred_calvo, y_pred_cabelloNegro, y_pred_gordito, y_pred_gafas, y_pred_masculino, y_pred_bigote, y_pred_sombrero, y_pred_joven = modelo(X_train)
            lossCalvo = criterioBin(y_pred_calvo, y_calvo)
            lossCabelloNegro = criterioBin(y_pred_cabelloNegro, y_cabelloNegro)
            lossGordito = criterioBin(y_pred_gordito, y_gordito)
            lossGafas = criterioBin(y_pred_gafas, y_gafas)
            lossMasculino = criterioBin(y_pred_masculino, y_masculino)
            lossBigote = criterioBin(y_pred_bigote, y_bigote)
            lossSombrero = criterioBin(y_pred_sombrero, y_sombrero)
            lossJoven = criterioBin(y_pred_joven, y_joven)
            lossTotal = lossCalvo + lossCabelloNegro + lossGordito + lossGafas + lossMasculino + lossBigote + lossSombrero + lossJoven
            perdida_acumulada += lossTotal.item()
            optimizador.zero_grad()
            lossTotal.backward()
            optimizador.step()                        
            print(f"Epoca {i+1}/{hpEpocas} - Item {j}/{totalIteraciones} - Perdida: {lossTotal.item():.4f}")
        perdida_promedio = perdida_acumulada / totalIteraciones
        scheduler.step(perdida_promedio)
        if perdida_promedio < mejor_perdida:
            mejor_perdida = perdida_promedio
            torch.save(modelo.state_dict(), f"CNN_6C3P3FC8S_CalvoToJoven_{perdida_promedio:.4f}.pt")

print("8. Probar el Modelo (Predecir) y Calcular el Score")
modelo.eval()
num_correct = 0
num_samples = 0
with torch.no_grad():
    for j,(X,y_calvo, y_cabelloNegro, y_gordito, y_gafas, y_masculino, y_bigote, y_sombrero, y_joven) in enumerate(loaderTest):
        X_test = X.to(device)

        y_calvo = y_calvo.to(device)        
        y_cabelloNegro = y_cabelloNegro.to(device)
        y_gordito = y_gordito.to(device)
        y_gafas = y_gafas.to(device)        
        y_masculino = y_masculino.to(device)
        y_bigote = y_bigote.to(device)        
        y_sombrero = y_sombrero.to(device)
        y_joven = y_joven.to(device)        
        
        y_pred_calvo, y_pred_cabelloNegro, y_pred_gordito, y_pred_gafas, y_pred_masculino, y_pred_bigote, y_pred_sombrero, y_pred_joven = modelo(X_test)
        predicCalvo = (y_pred_calvo > 0.5).float().squeeze().int()
        predicCabelloNegro = (y_pred_cabelloNegro > 0.5).float().squeeze().int()
        predicGordito = (y_pred_gordito > 0.5).float().squeeze().int()
        predicGafas = (y_pred_gafas > 0.5).float().squeeze().int()
        predicMasculino = (y_pred_masculino > 0.5).float().squeeze().int()
        predicBigote = (y_pred_bigote > 0.5).float().squeeze().int()
        predicSombrero = (y_pred_sombrero > 0.5).float().squeeze().int()
        predicJoven = (y_pred_joven > 0.5).float().squeeze().int()
        num_correct += ((predicCalvo == y_calvo) & (predicCabelloNegro == y_cabelloNegro) & (predicGordito == y_gordito) & (predicGafas == y_gafas) & (predicMasculino == y_masculino) & (predicBigote == y_bigote) & (predicSombrero == y_sombrero) & (predicJoven == y_joven)).sum().item()
        num_samples += predicCalvo.size(0)
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