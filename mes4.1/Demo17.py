import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import joblib
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo17.ui", self)
        txtArchivo = self.findChild(QtWidgets.QLabel, "txtArchivo")
        btnAbrir = self.findChild(QtWidgets.QPushButton, "btnAbrir")
        btnClasificar = self.findChild(QtWidgets.QPushButton, "btnClasificar")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblImagen = self.findChild(QtWidgets.QLabel, "lblImagen")
        self.lblPrediccion = self.findChild(QtWidgets.QLabel, "lblPrediccion")
        btnAbrir.clicked.connect(self.abrirArchivo)
        btnClasificar.clicked.connect(self.clasificarPersona)
        btnNuevo.clicked.connect(self.nuevo)
        self.modelo = joblib.load("Sklearn_LFW_People_PCA.pkl")
        self.scaler = joblib.load("Scaler_LFW_People.pkl")
        self.pca = joblib.load("PCA_LFW_People.pkl")
        with open("lfw_people.txt", "r") as file:
            self.personas = [x.replace("\n","") for x in file.readlines()]
    
    def abrirArchivo(self):
        dlg = QFileDialog()
        dlg.setDirectory(r"C:\Data\Python\2026_01_DAIAP\Imagenes\LFW_People")
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
    
    def clasificarPersona(self):        
        imgResize = cv2.resize(self.imagenGris, (62, 47))
        imagen = imgResize.flatten()
        print("Shape Imagen: ", imagen.shape)        
        X_test = imagen.reshape(1, -1)

        X_test_scaled = self.scaler.transform(X_test)
        print("Shape X_test: ", X_test_scaled.shape)

        X_test_pca = self.pca.transform(X_test_scaled)
        print("Shape X_test_pca: ", X_test_pca.shape)

        y_predecido = self.modelo.predict(X_test_pca)
        print(y_predecido)
        indice = y_predecido[0]
        nombre = self.personas[indice]
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