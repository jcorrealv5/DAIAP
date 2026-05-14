from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

print("Demo 08: Trabajar con DataSets y DataLoaders para CIFAR-100")

print("1. Cargar el DataSet CIFAR100")
dstTrain = CIFAR100(root="../../DataSets",train=True,download=True,transform=ToTensor())
print(f"Total de Muestras: {len(dstTrain)}")
clases = dstTrain.classes

print("2. Crear el DataLoader a partir del DataSet CIFAR100")
loaderTrain = DataLoader(dstTrain, batch_size=32)

print("3. Crear un iterador para manejar los datos")
iterTrain = iter(loaderTrain)

print("4. Cargar Lotes de Datos")
X_Lote1, y_Lote1 = next(iterTrain)
X_Lote2, y_Lote2 = next(iterTrain)
X_Lote3, y_Lote3 = next(iterTrain)

print("5. Visualizar la Primera Imagen del Primer Lote")
tensorImagen = X_Lote1[0]
arrayImagen = (tensorImagen.permute(1,2,0).detach().numpy()*255).astype(int)
etiqueta = y_Lote1[0]
print(f"Shape tensorImagen: {tensorImagen.shape}")
print(f"Shape arrayImagen: {arrayImagen.shape}")
print(f"Etiqueta: {etiqueta}")
plt.imshow(arrayImagen)
plt.title(f"Objeto: {clases[etiqueta]}")
plt.show()