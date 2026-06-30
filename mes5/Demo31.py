import sys
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
import torch, cv2
from modDL import CNN_4C4P
from torchvision.transforms import v2
import torchvision.transforms as transforms
import numpy as np

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo31.ui", self)
        btnActivarCamara = self.findChild(QtWidgets.QPushButton, "btnActivarCamara")
        self.lblVideo = self.findChild(QtWidgets.QLabel, "lblVideo")
        btnActivarCamara.clicked.connect(self.activarCamara)

        #Cargar el Modelo previamente entrenado
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = CNN_4C4P(142, 1, 1).to(self.device)
        with open('CNN_4C4P_Pelos_0.0660.pt', 'rb') as file: 
            self.modelo.load_state_dict(torch.load(file, map_location=self.device, weights_only=True))
        archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
        self.clasificador = cv2.CascadeClassifier(archivoHaar)
        self.imagen = None

    def activarCamara(self):
        thread = WorkerVideo(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarImagen)
        thread.start()
    
    def mostrarImagen(self, imagen):
        alto = imagen.shape[0]
        ancho = imagen.shape[1]
        #print(f"ancho: {ancho}, alto: {alto}")
        qImagen = QImage(imagen, ancho, alto, 3 * ancho, QImage.Format_BGR888)
        pixmap = QPixmap(qImagen)
        self.lblVideo.setPixmap(pixmap)
        
    def mostrarRpta(self, rpta):
        dlg = QMessageBox()
        dlg.setText(rpta)
        dlg.exec()

class WorkerVideo(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, parent):
        super(WorkerVideo, self).__init__(parent)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            v2.Resize((142,142)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.Grayscale(num_output_channels=1),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5], std=[0.5])
        ])
        self.modelo = CNN_4C4P(142, 1, 1).to(self.device)
        with open('CNN_4C4P_Pelos_0.0660.pt', 'rb') as file: 
            self.modelo.load_state_dict(torch.load(file, map_location=self.device, weights_only=True))
        self.modelo.eval()
        archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
        self.clasificador = cv2.CascadeClassifier(archivoHaar)

    def run(self):
        rpta = ""
        cap = cv2.VideoCapture(0, 700)
        if(cap.isOpened()):
            while True:
                rpta, imagen = cap.read()
                if(rpta):
                    caras = self.clasificador.detectMultiScale(imagen, scaleFactor=1.05, minNeighbors=5, minSize=(70,70),flags=cv2.CASCADE_SCALE_IMAGE)
                    for(x,y,w,h) in caras:
                        cara = imagen[y:y+h, x:x+w]
                        cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,0,255),4)
                        imgResize = cv2.resize(cara, (142, 142))
                        imgGris = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
                        imgTensor = self.transform(imgGris)
                        X = imgTensor.unsqueeze(0).to(self.device)    
                        y_predict = self.modelo(X)
                        prediccion = (torch.sigmoid(y_predict) > 0.5).float().squeeze().long()
                        pelo = ("Con Pelo" if prediccion==1 else "Calvo")
                        cv2.putText(imagen, pelo, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=4)
                    self.progreso.emit(imagen)
                else:
                    rpta = "Ocurrio un error en la captura"
                    break
        else:
            rpta = "No existe la camara web"
        self.finalizado.emit(rpta)
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Ventana()
    frm.show()
    sys.exit( app.exec_() )