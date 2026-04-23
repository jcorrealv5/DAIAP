import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import joblib
import numpy as np
import cv2

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo23.ui", self)
        txtArchivo = self.findChild(QtWidgets.QLabel, "txtArchivo")
        btnAbrir = self.findChild(QtWidgets.QPushButton, "btnAbrir")
        btnClasificar = self.findChild(QtWidgets.QPushButton, "btnClasificar")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblImagen = self.findChild(QtWidgets.QLabel, "lblImagen")
        self.lblPrediccion = self.findChild(QtWidgets.QLabel, "lblPrediccion")
        btnAbrir.clicked.connect(self.abrirArchivo)
        btnClasificar.clicked.connect(self.clasificarSexo)
        btnNuevo.clicked.connect(self.nuevo)
        self.modelo = joblib.load("Sklearn_Sexo.pkl")
        archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
        self.clasificador = cv2.CascadeClassifier(archivoHaar)
    
    def abrirArchivo(self):
        dlg = QFileDialog()
        dlg.setDirectory(r"C:\Users\ASUS\scikit_learn_data\lfw_home\lfw_funneled")
        dlg.setNameFilter(("Imagenes (*.png *.jpeg *.jpg)"))
        dlg.exec()
        archivos = dlg.selectedFiles()
        if(len(archivos)>0):
            self.txtArchivo.setText(archivos[0])
            imagen = cv2.imread(archivos[0])
            self.imagenGris = cv2.imread(archivos[0], 0)
            alto = imagen.shape[0]
            ancho = imagen.shape[1]
            qImagen = QImage(imagen, ancho, alto, 3 * ancho, QImage.Format_BGR888)
            pixmap = QPixmap(qImagen)
            self.lblImagen.setPixmap(pixmap)
    
    def clasificarSexo(self):
        caras = self.clasificador.detectMultiScale(self.imagenGris, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in caras:
            cara = self.imagenGris[y:y+h, x:x+w]
        nCaras = len(caras)
        if(nCaras>0):
            imgResize = cv2.resize(cara, (100, 100))
            imagen = imgResize.flatten()
            print("Shape Imagen: ", imagen.shape)        
            X_test = imagen.reshape(1, -1)
            y_predecido = self.modelo.predict(X_test)
            print(y_predecido)
            sexo = y_predecido[0]
            nombre = "Masculino" if sexo=="M" else "Femenino"
            self.lblPrediccion.setText(nombre)
    
    def nuevo(self):
        self.limpiarImagen()
    
    def limpiarImagen(self):
        self.lblPrediccion.setText("")
        self.lblImagen.setPixmap(QPixmap())

    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Ventana()
    frm.show()
    sys.exit( app.exec_() )