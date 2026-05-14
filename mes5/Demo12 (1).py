import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QByteArray, QBuffer
import numpy as np
import cv2
import torch
from modDL import MLP
import torchvision.transforms as transforms

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo12.ui", self)
        btnClasificar = self.findChild(QtWidgets.QPushButton, "btnClasificar")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblEntrada = self.findChild(QtWidgets.QLabel, "lblEntrada")
        self.lblPrediccion = self.findChild(QtWidgets.QLabel, "lblPrediccion")
        btnClasificar.clicked.connect(self.clasificarDigitos)
        btnNuevo.clicked.connect(self.nuevo)
        self.limpiarImagen()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = MLP(784, 50, 10).to(self.device)
        with open('MLP_MNIST.pt', 'rb') as file: 
            self.modelo.load_state_dict(torch.load(file, map_location=self.device, weights_only=True))  

    def clasificarDigitos(self):
        imgNdArray = self.convertirPixmapToNdArray(self.pixmap)
        #print("Shape arrayImagen: ", imgNdArray)
        #cv2.imwrite("Digito.png",imgNdArray)
        imgResize = cv2.resize(imgNdArray, (28, 28))
        transform = transforms.ToTensor()
        imgTensor = transform(imgResize)
        self.modelo.eval()
        with torch.no_grad():
            data = imgTensor.view(1,784).to(self.device)
            y_pred = self.modelo(data)
            print("y_pred: ", y_pred)            
            prediccion = torch.argmax(y_pred)
        self.lblPrediccion.setText(str(prediccion.item()))
    
    def nuevo(self):
        self.limpiarImagen()
    
    def limpiarImagen(self):
        self.lblPrediccion.setText("")
        self.pixmap = QPixmap(self.lblEntrada.width(), self.lblEntrada.height())
        self.pixmap.fill(Qt.GlobalColor.black)
        self.lblEntrada.setPixmap(self.pixmap)

    def mouseMoveEvent(self, e):
        x = e.x() - self.lblEntrada.x()
        y = e.y() - self.lblEntrada.y()
        grafico = QPainter(self.pixmap)
        grafico.setPen(QPen(Qt.white, 25, Qt.SolidLine))
        grafico.drawPoint(x,y)
        grafico.end()
        self.lblEntrada.setPixmap(self.pixmap)
    
    def convertirPixmapToNdArray(self, pixmap):
        qimage = pixmap.toImage()
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.WriteOnly)
        qimage.save(buffer, "PNG")    
        np_array = np.frombuffer(byte_array, np.uint8)
        ndarray = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
        gray=cv2.cvtColor(ndarray,cv2.COLOR_BGR2GRAY)
        return gray
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Ventana()
    frm.show()
    sys.exit( app.exec_() )