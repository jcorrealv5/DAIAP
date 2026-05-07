import torch

print("Demo 05: Cambiar el Tipo de Datos y la Forma de un Tensor")

print("1. Cambiar el Tipo de Datos de un Tensor")
tensorEnteros = torch.tensor([1,2,3])
print(f"Tensor Enteros: {tensorEnteros}")
print(f"Tipo Dato tensorEnteros: {tensorEnteros.dtype}")
tensorFlotantes = tensorEnteros.to(torch.float32)
print(f"Tensor Flotantes: {tensorFlotantes}")
print(f"Tipo Dato tensorFlotantes: {tensorFlotantes.dtype}")

print("2. Transposicion de un Tensor")
tensor34 = torch.tensor([[1,2,3,4],[0,2,4,6],[1,3,5,7]])
print(f"Tensor 3 Filas x 4 Cols: {tensor34}")
print(f"Shape tensor34: {tensor34.shape}")
tensor43 = torch.transpose(tensor34, 0, 1)
print(f"Tensor 4 Filas x 3 Cols: {tensor43}")
print(f"Shape tensor43: {tensor43.shape}")

print("3. Cambiar la Forma de un Tensor")
tensor62 = tensor43.reshape(6,2)
print(f"Tensor 6 Filas x 2 Cols: {tensor62}")
print(f"Shape tensor62: {tensor62.shape}")

print("4. Aumentar una dimension a un Tensor")
tensor162 = torch.unsqueeze(tensor62, 0)
print(f"Tensor 1 x 6 Filas x 2 Cols: {tensor162}")
print(f"Shape tensor162: {tensor162.shape}")

print("5. Eliminar una dimension a un Tensor")
tensor62 = torch.squeeze(tensor162, 0)
print(f"Tensor 6 Filas x 2 Cols: {tensor62}")
print(f"Shape tensor62: {tensor62.shape}")