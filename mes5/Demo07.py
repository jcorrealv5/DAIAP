from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

print("1. Cargando el DataSet MNIST")
dstTrain = MNIST(root="..\\..\\DataSets", train=True, download=True,transform=ToTensor())
print(f"Tipo objeto dstTrain: {type(dstTrain)}")
print(f"Total de Muestras: {len(dstTrain)}")

print("2. Creando el DataLoader del MNIST")
loaderTrain = DataLoader(dstTrain, batch_size=100)
print(f"Tipo objeto loaderTrain: {type(loaderTrain)}")

print("3. Crear un Iterador para Trabajar con Varios Lotes")
iterTrain = iter(loaderTrain)

def mostrarLote(lote):
    X_Lote,y_Lote = next(iterTrain)
    print(f"Shape {lote} X: {X_Lote.shape}")
    print(f"Shape {lote} y: {y_Lote.shape}")
    #print(f"Primer Digito del {lote} X: {X_Lote[0]}")
    print(f"Primer Digito del {lote} y: {y_Lote[0]}")
    return X_Lote,y_Lote

print("3. Cargando los Primeros Lotes de 100 muestras")
X_Lote1,y_Lote1 = mostrarLote("Lote 1")
X_Lote2,y_Lote2 = mostrarLote("Lote 2")
X_Lote3,y_Lote3 = mostrarLote("Lote 3")

def convertirTensorToArray(tensor):
    arreglo = (tensor.permute(1,2,0).detach().numpy() * 255).astype(int)
    print(f"Shape arreglo: {arreglo.shape}")
    return arreglo

def plotearArray(arreglo, titulo):
    plt.imshow(arreglo, cmap="gray")
    plt.title(titulo)
    plt.show()

print("4. Graficar el Primer Digito del Tercer Lote con su Etiqueta")
digitoTensor = X_Lote3[0]
etiqueta = y_Lote3[0]
print(f"Digito: {digitoTensor.shape}")
print(f"Etiqueta: {etiqueta}")
digitoArray = convertirTensorToArray(digitoTensor)
print(f"Bytes del Digito: {digitoArray}")
plotearArray(digitoArray,"Digito: " + str(etiqueta.item()))