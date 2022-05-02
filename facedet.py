import cv2
from facenet_pytorch import MTCNN,InceptionResnetV1
from torch.utils.data import DataLoader

import torch
import numpy as np
import mmcv
from torchvision import datasets,transforms
from PIL import Image, ImageDraw
from IPython import display
import sys
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

ear_cascade = cv2.CascadeClassifier('cascade.xml')
cascPath = 'facedet.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

mtcnn = MTCNN(device=device)

# data = datasets.ImageFolder('images',transform=transforms.ToTensor())

# loader = DataLoader(data)


# for img,lab in loader:
#     # print(img.squeeze(0).shape)
#     # plt.imshow(img.squeeze(0).permute(1,2,0).detach().cpu().numpy())
#     # plt.show()
#     pass
img = Image.open('images/faces/friends.jpg')

print(img.size)
boxes, _ = mtcnn.detect(img)
imgc = img.copy()
draw = ImageDraw.Draw(imgc)
for box in boxes:
    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
display.display(imgc)
plt.imshow(imgc)
plt.show()
