import torch, cv2
from modDL import CNN_4C4P
from torchvision.transforms import v2
import torchvision.transforms as transforms

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
modelo = CNN_4C4P(142, 1, 1).to(device)
with open('CNN_4C4P_Sexo_0.0460.pt', 'rb') as file: 
    modelo.load_state_dict(torch.load(file, map_location=device, weights_only=True))
imagen = cv2.imread("Personas.jpg")
caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
modelo.eval()
for(x,y,w,h) in caras:
    cara = imagen[y:y+h, x:x+w]
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,0,255),4)
    imgResize = cv2.resize(cara, (142, 142))
    imgGris = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
    imgTensor = transform(imgGris)
    print(f"Shape imgTensor: {imgTensor.shape}")
    X = imgTensor.unsqueeze(0).to(device)
    print(f"Shape X: {X.shape}")
    y_predict = modelo(X)
    prediccion = (torch.sigmoid(y_predict) > 0.5).float().squeeze().long()
    sexo = ("Hombre" if prediccion==1 else "Mujer")
    print(f"Sexo: {sexo}")
    cv2.putText(imagen, sexo, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=4)
cv2.imshow("Detectar Sexo", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()