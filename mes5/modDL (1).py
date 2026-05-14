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