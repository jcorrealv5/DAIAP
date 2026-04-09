import sys, cv2, os
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QListWidgetItem
from PyQt5.QtGui import QPixmap, QImage
sys.path.append("../00_Modulos")
from modWinPyQT import Imagen
from pathlib import Path
import numpy as np

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("dlgDemo72.ui", self)
        #Obtener los controles de la UI
        self.txtDirectorio = self.findChild(QtWidgets.QLineEdit, "txtDirectorio")
        btnAbrir = self.findChild(QtWidgets.QPushButton, "btnAbrir")
        self.lstArchivos = self.findChild(QtWidgets.QListWidget, "lstArchivos")
        self.lblNroObjetos = self.findChild(QtWidgets.QLabel, "lblNroObjetos")
        self.lblImagenOriginal = self.findChild(QtWidgets.QLabel, "lblImagenOriginal")        
        self.lblImagenObjetos = self.findChild(QtWidgets.QLabel, "lblImagenObjetos")
        self.lblNroArchivos = self.findChild(QtWidgets.QLabel, "lblNroArchivos")
        btnDetectarObjetos = self.findChild(QtWidgets.QPushButton, "btnDetectarObjetos")
        btnNuevo = self.findChild(QtWidgets.QPushButton, "btnNuevo")
        #Programar los eventos de los controles
        btnAbrir.clicked.connect(self.abrirDirectorio)
        btnDetectarObjetos.clicked.connect(self.detectarObjetos)
        btnNuevo.clicked.connect(self.nuevo)
        #Inicializar variables
        self.imagenOriginal = None
        archivoCategorias = r"C:\Data\Python\2026_01_DAIAP\Archivos\coco.names"
        self.clases = []
        with open(archivoCategorias, "r") as file:
            self.clases = [line.strip() for line in file.readlines()]
        archivoPesos = r"C:\Data\Python\2026_01_DAIAP\Archivos\yolov3.weights"
        archivoConfig = r"C:\Data\Python\2026_01_DAIAP\Archivos\yolov3.cfg"
        self.modelo = cv2.dnn.readNet(archivoPesos, archivoConfig)
        nombresCapas =  self.modelo.getLayerNames()
        self.capasSalida = [nombresCapas[i-1] for i in self.modelo.getUnconnectedOutLayers()]

    def abrirDirectorio(self):
        self.directorio = QFileDialog.getExistingDirectory(None, "Selecciona un Directorio")
        if(self.directorio is not None):
            self.txtDirectorio.setText(self.directorio)
            archivos = os.listdir(self.directorio)
            for archivo in archivos:
                ext = Path(archivo).suffix.lower()
                if(ext==".jpg" or ext==".jpeg" or ext==".png"):
                    item = QListWidgetItem(archivo)
                    self.lstArchivos.addItem(item)
            nArchivos = self.lstArchivos.count()
            self.lblNroArchivos.setText(str(nArchivos))

    def detectarObjetos(self):
        nArchivos = self.lstArchivos.count()        
        if(nArchivos>0):
            thread = WorkerImagen(self)
            thread.finalizado.connect(self.mostrarRpta)
            thread.progreso.connect(self.mostrarProgreso)
            thread.start()
        else:
            dlg=QMessageBox()
            dlg.setText("No hay imagenes que analizar")
            dlg.exec()
    
    def mostrarRpta(self, rpta):
        dlg=QMessageBox()
        dlg.setText(rpta)
        dlg.exec()
    
    def mostrarProgreso(self, imagenOriginal, imagenObjetos, nObjetos):
        Imagen.MostrarImagenColor(imagenOriginal, self.lblImagenOriginal)
        Imagen.MostrarImagenColor(imagenObjetos, self.lblImagenObjetos)
        self.lblNroObjetos.setText(str(nObjetos))

    def nuevo(self):
        self.txtArchivo.setText("")
        self.lblNroObjetos.setText("")
        self.imagenOriginal = None
        Imagen.LimpiarImagen(self.lblImagenOriginal)
        Imagen.LimpiarImagen(self.lblImagenObjetos)

class WorkerImagen(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(np.ndarray, np.ndarray, int)
    
    def __init__(self, parent):
        super(WorkerImagen, self).__init__(parent)
        self.directorio = parent.directorio
        self.lstArchivos = parent.lstArchivos
        self.modelo = parent.modelo
        self.capasSalida = parent.capasSalida
        self.clases = parent.clases
    
    def run(self):
        nArchivos = len(self.lstArchivos)
        total = 0
        for i in range(nArchivos):
            nombre = self.lstArchivos.item(i).text()
            print("Procesando: " + nombre)
            archivo = os.path.join(self.directorio, nombre)
            imagen = cv2.imread(archivo)            
            imagenObjetos = imagen.copy()
            ancho = imagen.shape[1]
            alto = imagen.shape[0]
            entrada = cv2.dnn.blobFromImage(imagen, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.modelo.setInput(entrada)
            salida = self.modelo.forward(self.capasSalida)
            class_ids = []
            confidences = []
            boxes = []
            c = 0
            for out in salida:
                for detection in out:
                    c+=1
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * ancho)
                        center_y = int(detection[1] * alto)
                        w = int(detection[2] * ancho)
                        h = int(detection[3] * alto)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            nDetecciones = len(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            colores = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
            diccionario = {}
            c = 0
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.clases[class_ids[i]])
                    if(not label in diccionario):
                        diccionario[label] = colores[c % len(colores)]
                        c=c+1
                    cv2.rectangle(imagenObjetos, (x, y), (x + w, y + h), diccionario[label], 2)
                    cv2.putText(imagenObjetos, label, (x, y + 30), font, 3, diccionario[label], 3)
            if(nDetecciones>0):
                rutaSalida = "C:/Data/Python/2026_01_DAIAP/Imagenes/TestSalidaV3"
                archivoSalida = os.path.join(rutaSalida, nombre)
                cv2.imwrite(archivoSalida, imagenObjetos)
                total+=1
            self.progreso.emit(imagen, imagenObjetos, nDetecciones)
        rpta = "Se copiaron " + str(total) + " Archivos con Objetos"
        self.finalizado.emit(rpta)
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Ventana()
    frm.show()
    sys.exit(app.exec_())