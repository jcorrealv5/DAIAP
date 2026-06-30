import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import torch, cv2
from modDL import CNN_4C4P
from torchvision.transforms import v2
import torchvision.transforms as transforms

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo29.ui", self)
        txtArchivo = self.findChild(QtWidgets.QLabel, "txtArchivo")
        btnAbrir = self.findChild(QtWidgets.QPushButton, "btnAbrir")
        btnClasificar = self.findChild(QtWidgets.QPushButton, "btnClasificar")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblImagen = self.findChild(QtWidgets.QLabel, "lblImagen")
        btnAbrir.clicked.connect(self.abrirArchivo)
        btnClasificar.clicked.connect(self.clasificarPelos)
        btnNuevo.clicked.connect(self.nuevo)
        #Cargar el Modelo previamente entrenado
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = CNN_4C4P(142, 1, 1).to(self.device)
        with open('CNN_4C4P_Pelos_0.0660.pt', 'rb') as file: 
            self.modelo.load_state_dict(torch.load(file, map_location=self.device, weights_only=True))
        archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
        self.clasificador = cv2.CascadeClassifier(archivoHaar)
        self.imagen = None
    
    def mostrarImagen(self):
        alto = self.imagen.shape[0]
        ancho = self.imagen.shape[1]
        qImagen = QImage(self.imagen, ancho, alto, 3 * ancho, QImage.Format_BGR888)
        pixmap = QPixmap(qImagen)
        self.lblImagen.setPixmap(pixmap)

    def abrirArchivo(self):
        dlg = QFileDialog()
        dlg.setDirectory(r"C:\Data\Python\2026_01_DAIAP\Imagenes\Pelos")
        dlg.setNameFilter(("Imagenes (*.png *.jpeg *.jpg)"))
        dlg.exec()
        archivos = dlg.selectedFiles()
        if(len(archivos)>0):
            self.txtArchivo.setText(archivos[0])
            self.imagen = cv2.imread(archivos[0])
            self.mostrarImagen()
    
    def clasificarPelos(self):
        if(self.imagen is not None):
            caras = self.clasificador.detectMultiScale(self.imagen, scaleFactor=1.1, minNeighbors=5, minSize=(70,70),flags=cv2.CASCADE_SCALE_IMAGE)
            self.modelo.eval()
            transform = transforms.Compose([
                v2.Resize((142,142)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(10),
                v2.Grayscale(num_output_channels=1),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])
            for(x,y,w,h) in caras:
                cara = self.imagen[y:y+h, x:x+w]
                cv2.rectangle(self.imagen,(x,y),(x+w,y+h),(0,0,255),4)
                #print(f"Ancho: {w} - Alto: {h}")
                imgResize = cv2.resize(cara, (142, 142))
                imgGris = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
                imgTensor = transform(imgGris)
                X = imgTensor.unsqueeze(0).to(self.device)    
                y_predict = self.modelo(X)
                prediccion = (torch.sigmoid(y_predict) > 0.5).float().squeeze().long()
                pelo = ("Pelos" if prediccion==1 else "Calvo")
                cv2.putText(self.imagen, pelo, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=4)
            self.mostrarImagen()
            self.imagen = None
        else:
            dlg = QMessageBox()
            dlg.setText("Seleccione primero una imagen a clasificar")
            dlg.exec()
    
    def nuevo(self):
        self.txtArchivo.setText("")
        self.lblImagen.setPixmap(QPixmap())
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Ventana()
    frm.show()
    sys.exit( app.exec_() )