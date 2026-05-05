import sys, cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn.functional import pairwise_distance

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo52.ui", self)
        #Definir controles como objetos
        self.cboTipoFiltro = self.findChild(QtWidgets.QComboBox, "cboTipoFiltro")
        self.txtArchivo = self.findChild(QtWidgets.QLineEdit, "txtArchivo")
        btnAbrir = self.findChild(QtWidgets.QPushButton, "btnAbrir")
        btnComparar = self.findChild(QtWidgets.QPushButton, "btnComparar")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblImagen1 = self.findChild(QtWidgets.QLabel, "lblImagen1")
        self.lblImagen2 = self.findChild(QtWidgets.QLabel, "lblImagen2")
        self.lblRespuesta = self.findChild(QtWidgets.QLabel, "lblRespuesta")
        #Programar Eventos
        btnAbrir.clicked.connect(self.abrirArchivo)
        btnComparar.clicked.connect(self.compararFotos)
        btnNuevo.clicked.connect(self.nuevo)
        #Llenar el Combo con 2 opciones: Imagen1 y Imagen2
        self.cboTipoFiltro.addItem("Imagen1")
        self.cboTipoFiltro.addItem("Imagen2")
        self.mtcnn = MTCNN()
        self.modelo = InceptionResnetV1(pretrained='vggface2').eval()

    
    def mostrarImagen(self, imagen, label):
        alto = imagen.shape[0]
        ancho = imagen.shape[1]
        qImagen = QImage(imagen, ancho, alto, 3 * ancho, QImage.Format_BGR888)
        pixmap = QPixmap(qImagen)
        label.setPixmap(pixmap)
    
    def abrirArchivo(self):
        dlg = QFileDialog()
        dlg.setDirectory(r"C:\Data\Python\2026_01_DAIAP\Imagenes\Empleados")
        dlg.setNameFilter(("Imagenes (*.png *.jpeg *.jpg)"))
        dlg.exec()
        archivos = dlg.selectedFiles()
        if(len(archivos)>0):
            self.txtArchivo.setText(archivos[0])
            if(self.cboTipoFiltro.currentIndex()==0):
                self.imagen1 = cv2.imread(archivos[0])
                self.mostrarImagen(self.imagen1, self.lblImagen1)
            else:
                self.imagen2 = cv2.imread(archivos[0])
                self.mostrarImagen(self.imagen2, self.lblImagen2)
    
    def compararFotos(self):
        rostros1 = self.mtcnn(self.imagen1)
        rostros2 = self.mtcnn(self.imagen2)
        if(rostros1 is not None and rostros2 is not None):
            embeddings1 = self.modelo(rostros1.unsqueeze(0))
            embeddings2 = self.modelo(rostros2.unsqueeze(0))
            distancia = pairwise_distance(embeddings1, embeddings2)
            valor = distancia.item()
            print(f"Distancia: {valor}")
            if valor < 0.8:
                self.lblRespuesta.setText("Es la misma persona")
            else:
                self.lblRespuesta.setText("Son personas diferentes")
        else:
            self.lblRespuesta.setText("No se detecto 1 o 2 rostros")
    
    def nuevo(self):
        self.txtArchivo.setText("")
        self.lblImagen1.setPixmap(QPixmap())
        self.lblImagen2.setPixmap(QPixmap())
        self.lblRespuesta.setText("")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Ventana()
    frm.show()
    sys.exit( app.exec_() )