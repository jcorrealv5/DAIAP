import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, numEntradas, numCapasOcultas, numClases):
        super(MLP, self).__init__()
        self.FCL1 = nn.Linear(numEntradas,numCapasOcultas)
        self.FCL2 = nn.Linear(numCapasOcultas,numClases)
    
    def forward(self, x):
        #print("MLP Shape x: ", x.shape)
        x = F.relu(self.FCL1(x))
        x = self.FCL2(x)
        return x
    
class CNN_2C1P(nn.Module):
    def __init__(self, sizeImagen, numClases):
        super(CNN_2C1P, self).__init__()
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.Pool = nn.MaxPool2d(2, 2)
        self.Dropout = nn.Dropout(0.25)
        size = int((sizeImagen - 2 - 2) / 2)
        entrada = 64 * size * size
        self.FCL1 = nn.Linear(entrada,1024)
        self.FCL2 = nn.Linear(1024,numClases)
    
    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = self.Dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.FCL1(x))
        x = self.FCL2(x)
        return x