import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QByteArray, QBuffer
import joblib
import numpy as np
import cv2

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo20.ui", self)
        btnClasificar = self.findChild(QtWidgets.QPushButton, "btnClasificar")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblEntrada = self.findChild(QtWidgets.QLabel, "lblEntrada")
        self.lblPrediccion = self.findChild(QtWidgets.QLabel, "lblPrediccion")
        btnClasificar.clicked.connect(self.clasificarCaracteres)
        btnNuevo.clicked.connect(self.nuevo)
        self.limpiarImagen()
        self.modelo = joblib.load("Sklearn_Caracteres.pkl")

    def clasificarCaracteres(self):
        imgNdArray = self.convertirPixmapToNdArray(self.pixmap)
        imgResize = cv2.resize(imgNdArray, (32, 32))
        imagen = imgResize.flatten()
        print(imagen.shape)
        imgs = np.array([imagen])
        y_predecido = self.modelo.predict(imgs)
        self.lblPrediccion.setText(y_predecido[0])
    
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
        grafico.setPen(QPen(Qt.white, 30, Qt.SolidLine))
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