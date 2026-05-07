import numpy as np
import torch
from torchvision.io import decode_image
import matplotlib.pyplot as plt

print("Demo 04: Creacion de Tensores en PyTorch")

print("1. Desde una Lista de Python")
lista = [10,20,30,40,50]
tensor1 = torch.tensor(lista)
print(f"Lista: {lista}")
print(f"Tensor desde lista: {tensor1}")
print(f"Shape Tensor lista: {tensor1.shape}")

print("2. Desde un Array de NumPy")
array = np.array([10,20,30,40,50])
tensor2 = torch.tensor(array)
print(f"Array de NumPy: {array}")
print(f"Shape Array: {array.shape}")
print(f"Tensor desde NumPy: {tensor2}")
print(f"Shape Tensor Array: {tensor2.shape}")

print("3. Desde una Imagen de Disco")
archivoImagen = "/Users/jhon.correal/Documents/Python/Shifu/DAIAP/mes5/Lena.png"
tensor3 = decode_image(archivoImagen, mode="RGB")
print(f"Imagen Decodificada: {tensor3}")
print(f"Shape Tensor Imagen: {tensor3.shape}")
array = tensor3.permute(2,1,0).numpy()
print(f"Shape Array: {array.shape}")
plt.imshow(array)
plt.show()

print("4. Tensor Relleno de Ceros y Unos")
tensor4 = torch.zeros(10)
print(f"Tensor de Ceros: {tensor4}")
print(f"Shape Tensor Ceros: {tensor4.shape}")
tensor5 = torch.ones(10)
print(f"Tensor de Unos: {tensor5}")
print(f"Shape Tensor Unos: {tensor5.shape}")

print("5. Tensor Numeros al Azar o Pseudo-Aleatorios")
tensor6 = torch.rand(10)
print(f"Tensor de Aleatorios: {tensor6}")
print(f"Shape Tensor Aleatorios: {tensor6.shape}")
