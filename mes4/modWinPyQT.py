from PyQt5.QtCore import QAbstractTableModel, Qt, QSize
from PyQt5.QtGui import QColor, QPixmap, QImage
from decimal import Decimal
from datetime import date
import os

class TablaModelo(QAbstractTableModel):
	def __init__(self, lista, formatoFilas=None, imagenes=None):
		QAbstractTableModel.__init__(self)
		self.lista = lista
		self.cabeceras = lista[0]
		self.formatoFilas = formatoFilas
		self.imagenes = imagenes

	def rowCount(self, parent=None):
		return len(self.lista)-1

	def columnCount(self, parent=None):        
		return len(self.cabeceras)

	def headerData(self, seccion, orientacion, rol):
		if(rol==Qt.DisplayRole):
			if(orientacion==Qt.Horizontal):
				return self.cabeceras[seccion]
			else:
				return seccion+1

	def data(self, celda, rol=Qt.DisplayRole):
		indiceFila = celda.row()
		indiceColumna = celda.column()
		valor = self.lista[indiceFila+1][indiceColumna]
		#print(f"fila: {indiceFila}, col: {indiceColumna}, valor: {valor}")
		if(rol==Qt.DisplayRole):
			if isinstance(valor, float) or isinstance(valor, Decimal):
				return "%.2f" % valor
			else:
				if isinstance(valor, date):
					return valor.strftime("%Y-%m-%d")
				else:
					return valor
		if(rol==Qt.TextAlignmentRole):
			if isinstance(valor, int) or isinstance(valor, Decimal):
				return Qt.AlignVCenter + Qt.AlignRight
			else:
				return Qt.AlignVCenter + Qt.AlignLeft
		if(self.formatoFilas is not None):
			c=0
			if("ColorFondoImpar" in self.formatoFilas):
				colorFondoImpar = self.formatoFilas["ColorFondoImpar"]
				c=c+1
			if("ColorTextoImpar" in self.formatoFilas):
				colorTextoImpar = self.formatoFilas["ColorTextoImpar"]
				c=c+1
			if("ColorFondoPar" in self.formatoFilas):
				colorFondoPar = self.formatoFilas["ColorFondoPar"]
				c=c+1
			if("ColorTextoPar" in self.formatoFilas):
				colorTextoPar = self.formatoFilas["ColorTextoPar"]
				c=c+1
			if(c==4):
				if(rol== Qt.BackgroundRole):
					if(indiceFila%2==0):
						return QColor(colorFondoImpar)
					else:
						return QColor(colorFondoPar)	
				if(rol== Qt.ForegroundRole):
					if(indiceFila%2==0):
						return QColor(colorTextoImpar)
					else:
						return QColor(colorTextoPar)
		if(rol==Qt.DecorationRole):
			if(self.imagenes is not None):
				if("tipo" in self.imagenes):
					tipo = self.imagenes["tipo"]
				else:
					tipo = "jpg"
				if(str(indiceColumna) in self.imagenes):
					ruta = self.imagenes[str(indiceColumna)]
					archivoImagen = os.path.join(ruta, str(valor) + "." + tipo)
					if(not os.path.isfile(archivoImagen)):
						archivoImagen = os.path.join(ruta, "No." + tipo)
					pix = QPixmap(archivoImagen)
					bmp = pix.scaled(80, 80, Qt.KeepAspectRatio)
					#print(f"Imagen Fila: {indiceFila} Col: {indiceColumna}")
					return bmp

class DataFrameModelo(QAbstractTableModel):
	def __init__(self, df, esPolar=False):
		QAbstractTableModel.__init__(self)
		self.df = df
		self.nFilas = df.shape[0]
		self.nCols = df.shape[1]
		self.esPolar = esPolar

	def rowCount(self, parent=None):
		return self.nFilas

	def columnCount(self, parent=None):        
		return self.nCols

	def headerData(self, indice, orientacion, rol):
		if(rol==Qt.DisplayRole):
			if(orientacion==Qt.Horizontal):
				return self.df.columns[indice]
			else:
				return indice+1

	def data(self, celda, rol=Qt.DisplayRole):
		indiceFila = celda.row()
		indiceColumna = celda.column()
		if(not self.esPolar):
			valor = self.df.values[indiceFila][indiceColumna]
		else:
			valor = self.df.item(indiceFila,indiceColumna)
		if(rol==Qt.DisplayRole):
			if(self.esPolar):
				tipo = str(self.df.schema.dtypes()[indiceColumna])
				if(tipo.startswith("Decimal")):
					return f"{valor:.2f}"
				else:
					return valor
			else:
				return valor

class Imagen():
	def MostrarImagenColor(imagen, label):
		alto = imagen.shape[0]
		ancho = imagen.shape[1]
		qImagen = QImage(imagen, ancho, alto, 3 * ancho, QImage.Format_BGR888)
		pixmap = QPixmap(qImagen)
		label.setPixmap(pixmap)

	def MostrarImagenGris(imagen, label):
		alto = imagen.shape[0]
		ancho = imagen.shape[1]
		qImagen = QImage(imagen, ancho, alto, QImage.Format_Grayscale8)
		pixmap = QPixmap(qImagen)
		label.setPixmap(pixmap)
	
	def LimpiarImagen(label):
		label.setPixmap(QPixmap())