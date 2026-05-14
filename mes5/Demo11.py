import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F
from modDL import MLP

print("Demo 11: Probar el Modelo MLP de Clasificacion de Digitos MNIST con Data de Pruebas del mismo DataSet")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1. Detectando el Dispositivo")
print(f"Dispositivo usado: {device}")

print("2. Cargando el DataSet MNIST")
dstTest = MNIST(root="..\\..\\DataSets", train=False, download=True,transform=ToTensor())
totalMuestras = len(dstTest)
print(f"Total de Muestras: {totalMuestras}")

hpTamanoLote=1
print("3. Creando el DataLoader del MNIST")
loaderTest = DataLoader(dstTest, batch_size=hpTamanoLote, shuffle=True)

hpCapasOcultas = 50
print("4. Crear un Modelo de Clasificacion")
modelo = MLP(784, hpCapasOcultas, 10).to(device)

print("5. Cargar los Pesos al Modelo desde el archivo PreEntrenado")
with open('Torch_MNIST.pt', 'rb') as file: 
     modelo.load_state_dict(torch.load(file, map_location=device, weights_only=True))     

print("6. Cargar el Digito a Probar")
X_test, y_test = next(iter(loaderTest))
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

print("7. Predecir el Digito")
modelo.eval()
with torch.no_grad():
     data = X_test.view(1,784).to(device)
     print("data: ", data.shape)
     y_pred = modelo(data)
print(f"y_pred: {y_pred}")
_,prediccion = torch.max(y_pred, 1)

print("Resultados de la Prediccion del Digito")
print(f"Digito Real: {y_test.item()}")
print(f"Digito Predecido: {prediccion.item()}")
