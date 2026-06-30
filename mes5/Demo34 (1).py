import torch, cv2
import torch.nn.functional as F
from modDL import CNN_6C3P3FC2S
from torchvision.transforms import v2
import torchvision.transforms as transforms

print("Demo 34: Detectar el sexo y raza en un Video en Tiempo Real usando 1 solo Modelo")

archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)
sizeImagen = 156
sizeRaza = 5
transform = transforms.Compose([
    v2.Resize((sizeImagen,sizeImagen)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelo = CNN_6C3P3FC2S(sizeRaza, sizeImagen).to(device)
with open('CNN_6C3P3FC2S_SexoRaza_0.9574.pt', 'rb') as file:
    modelo.load_state_dict(torch.load(file, map_location=device, weights_only=True))
modelo.eval()

cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    while True:
        rpta, imagen = cap.read()
        if(rpta):
            caras = clasificador.detectMultiScale(imagen, scaleFactor=1.05, minNeighbors=5, minSize=(70,70),flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in caras:
                cara = imagen[y:y+h, x:x+w]
                cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,0,255),4)
                imgResize = cv2.resize(cara, (sizeImagen, sizeImagen))
                imgTensor = transform(imgResize)
                X = imgTensor.unsqueeze(0).to(device)
                y_pred_sexo, y_pred_raza = modelo(X)
                predicSexo = (torch.sigmoid(y_pred_sexo) > 0.5).int()
                predicRaza = torch.argmax(F.log_softmax(y_pred_raza, dim=1), dim=1)
                sexo = ("Hombre" if predicSexo==1 else "Mujer")
                raza = ("Blanca" if predicRaza==0 else "Negra" if predicRaza==1 else "Asiatica" if predicRaza==2 else "India" if predicRaza==3 else "Otros")
                cv2.putText(imagen, sexo + " " + raza, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=4)
            cv2.imshow("Video", imagen)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Video Finalizado")
else:
    print("No tiene una Camara Web Conectada")