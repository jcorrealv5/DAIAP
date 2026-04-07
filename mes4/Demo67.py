import cv2

print("Demo 67: App de Consola para Detección de Marcas Faciales en una Imagen de Disco")
archivoImagen = r"C:\Data\Python\2026_01_DAIAP\Imagenes\Clasicos\Jorge_Profesor_Pedro.jpeg"
imagen = cv2.imread(archivoImagen)

archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)
caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

archivoModeloLBF = r"C:\Data\Python\2026_01_DAIAP\Archivos\lbfmodel.yaml"
lbf = cv2.face.createFacemarkLBF()
lbf.loadModel(archivoModeloLBF)
_, marcasFaciales = lbf.fit(imagen, caras)

for marca in marcasFaciales:
    print("marca: ", marca)
    for x,y in marca[0]:
        cv2.circle(imagen, (int(x), int(y)), 1, (0, 255, 0), 2)

nCaras = len(caras)
imagen = cv2.resize(imagen, (600,500))
cv2.imshow("Rostros Detectados: " + str(nCaras), imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()