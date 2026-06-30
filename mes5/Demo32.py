import torch, cv2
from modDL import CNN_4C4P
from torchvision.transforms import v2
import torchvision.transforms as transforms

print("Demo 32: Detectar el sexo y pelos en un Video en Tiempo Real usando 2 Modelos")

archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)
transform = transforms.Compose([
    v2.Resize((142,142)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(10),
    v2.Grayscale(num_output_channels=1),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modeloSexo = CNN_4C4P(142, 1, 1).to(device)
with open('CNN_4C4P_Sexo_0.0183.pt', 'rb') as file: 
    modeloSexo.load_state_dict(torch.load(file, map_location=device, weights_only=True))
modeloSexo.eval()

modeloPelo = CNN_4C4P(142, 1, 1).to(device)
with open('CNN_4C4P_Pelos_0.0660.pt', 'rb') as file: 
    modeloPelo.load_state_dict(torch.load(file, map_location=device, weights_only=True))
modeloPelo.eval()

cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    while True:
        rpta, imagen = cap.read()
        if(rpta):
            caras = clasificador.detectMultiScale(imagen, scaleFactor=1.05, minNeighbors=5, minSize=(70,70),flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in caras:
                cara = imagen[y:y+h, x:x+w]
                cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,0,255),4)
                imgResize = cv2.resize(cara, (142, 142))
                imgGris = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
                imgTensor = transform(imgGris)
                X = imgTensor.unsqueeze(0).to(device)    
                y_predict_sexo = modeloSexo(X)
                y_predict_pelo = modeloPelo(X)
                prediccionSexo = (torch.sigmoid(y_predict_sexo) > 0.5).float().squeeze().long()
                prediccionPelo = (torch.sigmoid(y_predict_pelo) > 0.5).float().squeeze().long()
                sexo = ("Hombre" if prediccionSexo==1 else "Mujer")
                pelo = ("Con Pelo" if prediccionPelo==1 else "Calvo")
                if(sexo=="Mujer"):
                    pelo="Calva"
                cv2.putText(imagen, sexo + " " + pelo, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=4)
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