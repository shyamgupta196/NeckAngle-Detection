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

mtcnn = MTCNN(device=device,keep_all=False)

img = Image.open('images/faces/friends.jpg')

# cap = cv2.VideoCapture(0)




print(img.size)
boxes, _ = mtcnn.detect(img)
imgc = img.copy()
draw = ImageDraw.Draw(imgc)
for box in boxes:
    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)
plt.imshow(imgc)
plt.show()

imgarr = np.array(imgc)
ear = ear_cascade.detectMultiScale(imgarr,minNeighbors=5,scaleFactor=1.15)

for (x,y,w,h) in ear:
    cv2.rectangle(imgarr, (x,y), (x+w,y+h), (255,0,0), 3)

cv2.imshow('img',imgarr)
plt.imshow(imgarr)
plt.show()
