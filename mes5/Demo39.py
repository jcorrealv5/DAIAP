import torchvision
from torchvision.models import list_models

listaModelosTodo = list_models()
listaModelosVision = list_models(module=torchvision.models)
print(f"listaModelosTodo {len(listaModelosTodo)}:")
with open("Pytorch_Modelos_Todos.txt", "w") as file1:
    for i,modelo in enumerate(listaModelosTodo):
        print(f"{i+1}: {modelo}")
        file1.write(f"{i+1}: {modelo}\n")
print(f"listaModelosVision {len(listaModelosVision)}:")
with open("Pytorch_Modelos_Vision.txt", "w") as file2:
    for i,modelo in enumerate(listaModelosVision):
        print(f"{i+1}: {modelo}")
        file2.write(f"{i+1}: {modelo}\n")
print("Lista de Modelos fue grabada a disco")