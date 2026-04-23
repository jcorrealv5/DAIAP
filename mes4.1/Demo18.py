import sys, os
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QByteArray, QBuffer
import numpy as np
import cv2

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo18.ui", self)
        self.txtCaracter = self.findChild(QtWidgets.QLineEdit, "txtCaracter")
        self.txtUsuario = self.findChild(QtWidgets.QLineEdit, "txtUsuario")
        btnGuardar = self.findChild(QtWidgets.QPushButton, "btnGuardar")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblCaracter = self.findChild(QtWidgets.QLabel, "lblCaracter")
        btnGuardar.clicked.connect(self.guardarCaracter)
        btnNuevo.clicked.connect(self.nuevo)
        self.limpiarImagen()

    def guardarCaracter(self):
        rutaDataSet = "C:/Data/Python/2026_01_DAIAP/DataSets/Caracteres/"
        rutaCaracter = rutaDataSet + self.txtCaracter.text()
        if(not os.path.isdir(rutaCaracter)):
            os.makedirs(rutaCaracter)
        archivos = os.listdir(rutaCaracter)
        usuario = self.txtUsuario.text()
        c=1
        for archivo in archivos:
            if(archivo.startswith(usuario)):
                c=c+1
        nombreArchivo = usuario + str(c) + ".png"
        archivoCaracter = os.path.join(rutaCaracter, nombreArchivo)
        imgNdArray = self.convertirPixmapToNdArray(self.pixmap)
        imgResize = cv2.resize(imgNdArray, (32, 32))
        print("Shape imagen: ", imgResize.shape)
        cv2.imwrite(archivoCaracter, imgResize)
        dlg = QMessageBox()
        dlg.setText("Se guardo el archivo: " + nombreArchivo)
        dlg.exec()
    
    def nuevo(self):
        self.limpiarImagen()
    
    def limpiarImagen(self):
        self.pixmap = QPixmap(self.lblCaracter.width(), self.lblCaracter.height())
        self.pixmap.fill(Qt.GlobalColor.black)
        self.lblCaracter.setPixmap(self.pixmap)

    def mouseMoveEvent(self, e):
        x = e.x() - self.lblCaracter.x()
        y = e.y() - self.lblCaracter.y()
        grafico = QPainter(self.pixmap)
        grafico.setPen(QPen(Qt.white, 30, Qt.SolidLine))
        grafico.drawPoint(x,y)
        grafico.end()
        self.lblCaracter.setPixmap(self.pixmap)
    
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