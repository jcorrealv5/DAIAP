from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt

print("Demo 09: Trabajar con DataSets y DataLoaders para GTSRB")

print("1. Cargar el DataSet GTSRB")
transform = T.Compose([T.ToTensor(),T.Resize((30, 30))])
dstTrain = GTSRB(root="../../DataSets",download=True,transform=transform)
print(f"Total de Muestras: {len(dstTrain)}")

print("2. Crear el DataLoader a partir del DataSet GTSRB")
loaderTrain = DataLoader(dstTrain, batch_size=32, shuffle=True)

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
plt.title(f"Objeto: {etiqueta}")
plt.show()