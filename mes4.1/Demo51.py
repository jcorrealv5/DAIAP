import os, cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

print("Demo 51: Ingresar la ruta de una foto y muestre su embedding")
archivo = input("Archivo con una foto: ")
if(os.path.isfile(archivo)):
    imagenPIL = Image.open(archivo)
    print("tipo de objeto: ", type(imagenPIL))
    #PIL.JpegImagePlugin.JpegImageFile
    imagenNP = cv2.imread(archivo)    
    print("Shape imagen: ", imagenNP.shape)
    mtcnn = MTCNN()
    modelo = InceptionResnetV1(pretrained='vggface2').eval()
    rostro = mtcnn(imagenPIL)
    print("Rostro: ", rostro)
    print("Shape rostro: ", rostro.shape)
    rostros = rostro.unsqueeze(0)
    print("Shape rostros: ", rostros.shape)
    embeddings = modelo(rostros)
    print("Shape embeddings: ", embeddings.shape)
    #print("Unico Embedding: ", embeddings[0])
    rostroNP = rostro.permute(2,1,0).numpy() * 255
    rostroNP = cv2.cvtColor(rostroNP, cv2.COLOR_BGR2RGB)
    rostroNP = cv2.rotate(rostroNP, cv2.ROTATE_90_CLOCKWISE)
    print("Shape rostroNP: ", rostroNP.shape)
    cv2.imwrite("Cara.png", rostroNP)
    print("Se grabo el archivo con la cara en disco")
else:
    print("Archivo ingresado No existe")