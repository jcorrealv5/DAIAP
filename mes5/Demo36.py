import torch, cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F
from modDL import DatasetCelebABin8,CNN_6C3P3FC8S

print("Demo 36: Clasificar los 8 atributos de una persona en una Foto")

archivoHaar = r"C:\Data\Python\2026_01_DAIAP\Archivos\haarcascade_frontalface_default.xml"
clasificador = cv2.CascadeClassifier(archivoHaar)
sizeImagen = 156
transform = transforms.Compose([
    v2.Resize((sizeImagen,sizeImagen)),
    v2.Grayscale(num_output_channels=1),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelo = CNN_6C3P3FC8S(sizeImagen).to(device)
with open('CNN_6C3P3FC8S_CalvoToJoven_1.1736.pt', 'rb') as file:
    modelo.load_state_dict(torch.load(file, map_location=device, weights_only=True))
modelo.eval()

archivoImagen = r"C:\Data\Python\2026_01_DAIAP\Imagenes\Clasicos\Turing_Bengio_Hinton_LeCun.png"
imagenColor = cv2.imread(archivoImagen)
imagen = cv2.imread(archivoImagen, 0)
caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
if(len(caras)>0):
    for(x,y,w,h) in caras:
        cara = imagen[y:y+h, x:x+w]
        cara = cv2.resize(cara, (sizeImagen, sizeImagen))
        caraColor = imagenColor[y:y+h, x:x+w]
        cv2.rectangle(imagenColor,(x,y),(x+w,y+h),(0,0,255),4)
        X_test = transform(cara).unsqueeze(0).to(device)
        print(f"Shape X_test: {X_test.shape}")
        y_pred_calvo, y_pred_cabelloNegro, y_pred_gordito, y_pred_gafas, y_pred_masculino, y_pred_bigote, y_pred_sombrero, y_pred_joven = modelo(X_test)
        predicCalvo = (y_pred_calvo > 0.5).float().squeeze().int()
        predicCabelloNegro = (y_pred_cabelloNegro > 0.5).float().squeeze().int()
        predicGordito = (y_pred_gordito > 0.5).float().squeeze().int()
        predicGafas = (y_pred_gafas > 0.5).float().squeeze().int()
        predicMasculino = (y_pred_masculino > 0.5).float().squeeze().int()
        predicBigote = (y_pred_bigote > 0.5).float().squeeze().int()
        predicSombrero = (y_pred_sombrero > 0.5).float().squeeze().int()
        predicJoven = (y_pred_joven > 0.5).float().squeeze().int()
        print(f"predicCalvo: {predicCalvo}")
        print(f"predicCabelloNegro: {predicCabelloNegro}")
        print(f"predicGordito: {predicGordito}")
        print(f"predicGafas: {predicGafas}")
        print(f"predicMasculino: {predicMasculino}")
        print(f"predicBigote: {predicBigote}")
        print(f"predicSombrero: {predicSombrero}")
        print(f"predicJoven: {predicJoven}")
cv2.imshow("8 Caracteristicas", imagenColor)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
