import package
from model import YOLOv1
from torchinfo import summary

model = YOLOv1(num_classes=20, num_anchors=2)
summary(model, input_size=(1, 3, 448, 448))

print("[+] Summary Genrated")