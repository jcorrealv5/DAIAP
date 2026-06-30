import torch, os, cv2
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class MLP(nn.Module):
    def __init__(self, numEntradas, numCapasOcultas, numClases):
        super(MLP, self).__init__()
        self.FCL1 = nn.Linear(numEntradas,numCapasOcultas)
        self.FCL2 = nn.Linear(numCapasOcultas,numClases)
    
    def forward(self, x):
        x = F.relu(self.FCL1(x))
        x = self.FCL2(x)
        return x
    
class CNN_2C1P(nn.Module):
    def __init__(self, sizeImagen, numCanales, numClases):
        super(CNN_2C1P, self).__init__()
        self.Conv1 = nn.Conv2d(numCanales, 32, kernel_size=3)
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

class CNN_3C3P(nn.Module):
    def __init__(self, sizeImagen, numCanales, numClases):
        super(CNN_3C3P, self).__init__()
        self.Conv1 = nn.Conv2d(numCanales, 32, kernel_size=3)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.Pool = nn.MaxPool2d(2, 2)
        self.Dropout = nn.Dropout(0.25)
        size = int((sizeImagen - 2) / 2)
        size = int((size - 2) / 2)
        size = int((size - 2) / 2)
        entrada = 128 * size * size
        self.FCL1 = nn.Linear(entrada,1024)
        self.FCL2 = nn.Linear(1024,numClases)
    
    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.Pool(x)
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = F.relu(self.Conv3(x))
        x = self.Pool(x)
        x = self.Dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.FCL1(x))
        x = self.FCL2(x)
        return x

class CNN_4C4P(nn.Module):
    def __init__(self, sizeImagen, numCanales, numClases):
        super(CNN_4C4P, self).__init__()
        self.Conv1 = nn.Conv2d(numCanales, 32, kernel_size=3)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.Conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.Pool = nn.MaxPool2d(2, 2)
        self.Dropout = nn.Dropout(0.25)
        size = int((sizeImagen - 2) / 2)
        size = int((size - 2) / 2)
        size = int((size - 2) / 2)
        size = int((size - 2) / 2)
        entrada = 256 * size * size
        self.FCL1 = nn.Linear(entrada,1024)
        self.FCL2 = nn.Linear(1024,numClases)
    
    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.Pool(x)
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = F.relu(self.Conv3(x))
        x = self.Pool(x)
        x = F.relu(self.Conv4(x))
        x = self.Pool(x)
        x = self.Dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.FCL1(x))
        x = self.FCL2(x)
        return x

class DatasetBinCat(Dataset):
    def __init__(self, ruta, transform=None, separador="_"):
        self.archivos = []
        self.etiquetasSexo = []
        self.etiquetasRaza = []
        self.caras = []
        self.transform = transform
        archivos = os.listdir(ruta)
        archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
        clasificador = cv2.CascadeClassifier(archivoHaar)
        for nombreArchivo in archivos:
            archivo = os.path.join(ruta, nombreArchivo)
            self.archivos.append(archivo)
            edad = int(nombreArchivo.split(separador)[0])
            sexo = int(nombreArchivo.split(separador)[1])
            raza = int(nombreArchivo.split(separador)[2])
            if(edad>17):
                imagen = cv2.imread(archivo)
                caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                if(len(caras)>0):
                    wMax = 0
                    hMax = 0
                    caraReal = None
                    for(x,y,w,h) in caras:
                        cara = imagen[y:y+h, x:x+w]
                        if(w>wMax and h>hMax):
                            wMax = w
                            hMax = h
                            caraReal = cv2.resize(cara.copy(),(156,156))
                    self.etiquetasSexo.append(sexo)
                    self.etiquetasRaza.append(raza)
                    self.caras.append(caraReal)
                    #print(f"archivo: {nombreArchivo}")
    
    def __len__(self):
        return len(self.etiquetasSexo)
        
    def __getitem__(self, indice):
        if(self.transform is not None):
            imagenTensor = self.transform(self.caras[indice])
        etiquetaSexo = self.etiquetasSexo[indice]
        etiquetaRaza = self.etiquetasRaza[indice]
        #print(f"{indice}: sexo: {etiquetaSexo} - raza: {etiquetaRaza}")
        return imagenTensor, etiquetaSexo, etiquetaRaza

class CNN_6C3P3FC2S(nn.Module):
    def __init__(self, nClases, nSize=156):
        super(CNN_6C3P3FC2S, self).__init__()        
        self.Conv1 = nn.Conv2d(3, 32, kernel_size=3)
        nSize = nSize - 2
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        nSize = nSize - 2
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.Pool1 = nn.MaxPool2d(2, 2)
        nSize = nSize / 2
        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3)
        nSize = nSize - 2
        self.BatchNorm3 = nn.BatchNorm2d(128)
        self.Conv4 = nn.Conv2d(128, 128, kernel_size=3)
        nSize = nSize - 2
        self.Pool2 = nn.MaxPool2d(2, 2)
        nSize = nSize / 2
        self.Conv5 = nn.Conv2d(128, 256, kernel_size=3)
        nSize = nSize - 2
        self.BatchNorm4 = nn.BatchNorm2d(256)
        self.Conv6 = nn.Conv2d(256, 256, kernel_size=3)
        nSize = nSize - 2
        self.Pool3 = nn.MaxPool2d(2, 2)
        nSize = int(nSize / 2)
        self.FC1 = nn.Linear(256 * nSize * nSize, 1024)
        self.FC2 = nn.Linear(1024, 128)
        self.FC3A = nn.Linear(128, 1)
        self.FC3B = nn.Linear(128, nClases)
        self.Dropout1 = nn.Dropout(0.25)
        self.Dropout2 = nn.Dropout(0.5)
        self.nSize = nSize        
    def forward(self,x):
        x = F.relu(self.BatchNorm1(self.Conv1(x)))        
        x = self.Dropout1(x)
        x = F.relu(self.BatchNorm2(self.Conv2(x)))
        x = self.Dropout1(x)
        x = self.Pool1(x)
        x = F.relu(self.BatchNorm3(self.Conv3(x)))
        x = self.Dropout1(x)
        x = F.relu(self.BatchNorm3(self.Conv4(x)))
        x = self.Dropout1(x)
        x = self.Pool2(x)
        x = F.relu(self.BatchNorm4(self.Conv5(x)))
        x = self.Dropout1(x)
        x = F.relu(self.BatchNorm4(self.Conv6(x)))
        x = self.Pool3(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 256 * self.nSize * self.nSize)
        x = F.relu(self.FC1(x))
        x = self.Dropout1(x)
        x = F.relu(self.FC2(x))
        x = self.Dropout2(x)
        sexo = self.FC3A(x)
        raza = F.log_softmax(self.FC3B(x), dim=1)
        return sexo, raza