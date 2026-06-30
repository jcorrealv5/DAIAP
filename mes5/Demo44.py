import torch, cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.io.image import decode_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

print("Demo 44: Deteccion de Objetos en Tiempo Real")
pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos, box_score_thresh=0.9)
modelo.eval()
transformacion = pesos.transforms()

cap = cv2.VideoCapture(0, 700)
c = 0
if(cap.isOpened()):
    while True:
        rpta, imagen = cap.read()
        if(rpta):
            imagenArrayRGB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagenTensor = torch.from_numpy(imagenArrayRGB).permute(2, 0, 1)
            imagenTransformada = transformacion(imagenTensor)
            imagenesLote = imagenTransformada.unsqueeze(0)
            c = c + 1
            if(c==30):
                c=0
                prediccion = modelo(imagenesLote)[0]
                clases = [pesos.meta["categories"][i] for i in prediccion["labels"]]
                colores = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta", "lime", "teal", "lavender", "brown", "beige", "maroon", "mint", "olive", "coral", "navy", "grey"]
                listaColores = [colores[int(clase) % len(colores)] for clase in prediccion["labels"]]
                imagenCuadros = draw_bounding_boxes(imagenTransformada, boxes=prediccion["boxes"],labels=clases,colors=listaColores,width=4, font_size=50)
                imagen = to_pil_image(imagenCuadros.detach())
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