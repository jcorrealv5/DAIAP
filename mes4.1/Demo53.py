import sys, os, cv2
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
import numpy as np

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo53.ui", self)
        #Definir controles como objetos
        self.txtCantidad = self.findChild(QtWidgets.QLineEdit, "txtCantidad")
        self.txtUsuario = self.findChild(QtWidgets.QLineEdit, "txtUsuario")
        btnActivarCamara = self.findChild(QtWidgets.QPushButton, "btnActivarCamara")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        self.lblVideo = self.findChild(QtWidgets.QLabel, "lblVideo")
        self.pbrCaptura = self.findChild(QtWidgets.QProgressBar, "pbrCaptura")
        #Programar Eventos
        btnActivarCamara.clicked.connect(self.activarCamara)
        btnNuevo.clicked.connect(self.nuevo)
    
    def activarCamara(self):
        self.cantidad = int(self.txtCantidad.text())
        self.usuario = self.txtUsuario.text()
        self.pbrCaptura.setMaximum(self.cantidad)
        thread = WorkerVideo(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarProgreso)
        thread.start()

    def mostrarProgreso(self, imagen, valor):
        self.pbrCaptura.setValue(valor)
        alto = imagen.shape[0]
        ancho = imagen.shape[1]
        qImagen = QImage(imagen, ancho, alto, 3 * ancho, QImage.Format_BGR888)
        pixmap = QPixmap(qImagen)
        self.lblVideo.setPixmap(pixmap)
    
    def mostrarRpta(self, texto):
        dlg = QMessageBox()
        dlg.setText(texto)
        dlg.exec()
    
    def nuevo(self):
        self.pbrCaptura.setValue(0)
        self.txtCantidad.setText("")
        self.txtUsuario.setText("")
        self.lblVideo.setPixmap(QPixmap())

class WorkerVideo(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(np.ndarray, int)
    
    def __init__(self, parent):
        super(WorkerVideo, self).__init__(parent)
        self.cantidad = parent.cantidad
        self.usuario = parent.usuario
        self.ruta = "C:/Data/Python/2026_01_DAIAP/Imagenes/Alumnos/" + self.usuario
        if(not os.path.isdir(self.ruta)):
            os.makedirs(self.ruta)

    def run(self):
        archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
        clasificador = cv2.CascadeClassifier(archivoHaar)
        cap = cv2.VideoCapture(0, 700)
        if(cap.isOpened()):
            c=0
            while c<self.cantidad:
                rpta, imagen = cap.read()
                if(rpta):
                    caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                    nCaras = len(caras)
                    if(nCaras>0):
                        margen=20
                        for(x,y,w,h) in caras:
                            cara = imagen[y-margen:y+h+margen,x-margen:x+w+margen].copy()
                            cv2.rectangle(imagen, (x,y), (x+w, y+h), (0,255,0), 3)
                            c+=1
                            nombre = self.usuario + "_" + str(c) + ".png"
                            archivo = os.path.join(self.ruta, nombre)
                            cv2.imwrite(archivo, cara)                            
                        self.progreso.emit(imagen, c)
                    key = cv2.waitKey(1)
                    if(key==ord("s")):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.finalizado.emit("Video Finalizado")
        else:
            print("No tiene una Camara Web Conectada")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Ventana()
    frm.show()
    sys.exit( app.exec_() )