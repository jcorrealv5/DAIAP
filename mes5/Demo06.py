import torch

print("Demo 06: Operaciones Matematicas con Tensores")
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[1,3,5],[2,4,6]])
tensor3 = torch.tensor([[1,3],[5,2],[4,6]])
print(f"Tensor1: {tensor1}")
print(f"Tensor2: {tensor2}")
print(f"Tensor3: {tensor3}")

print("1. Multiplicacion de Tensores x Elementos")
tensorMultiply = torch.multiply(tensor1, tensor2)
print(f"Tensor Multiply: {tensorMultiply}")

print("2. Multiplicacion de Tensores Matricial")
tensorMatmul = torch.matmul(tensor1, tensor3)
print(f"Tensor Matmul: {tensorMatmul}")

print("3. Multiplicacion Producto Punto de un Tensor1D")
tensor4=torch.tensor([1,3,5,7,9])
tensor5=torch.tensor([2,4,6,8,10])
tensorDotProduct = torch.dot(tensor4, tensor5)
print(f"Tensor DotProduct: {tensorDotProduct}")